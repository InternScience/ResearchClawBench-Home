"""DIDS-MFL evaluation - optimized for speed."""
import torch, torch.nn.functional as F
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import load_data, extract_flat_features, statistical_disentanglement, compute_feature_importance
from gnn_model import build_static_graphs, DIDS_MFL, EdgeClassifier
from few_shot import PrototypeNetwork, create_few_shot_episodes, evaluate_open_set
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings; warnings.filterwarnings('ignore')

SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)

def main():
    print("="*60+"\nDIDS-MFL Evaluation\n"+"="*60)
    
    # 1. Data
    print("\n[1] Loading data...")
    data=load_data()
    features,labels,attacks,timestamps,src,dst=extract_flat_features(data)
    print(f"  Flows: {len(features)}, Benign: {(labels==0).sum()}, Attack: {(labels==1).sum()}, Classes: {len(set(attacks))}")

    # 2. Statistical Disentanglement
    print("\n[2] Statistical disentanglement (PCA+ICA)...")
    X_ica,scaler,pca,ica=statistical_disentanglement(features,n_components=20)
    print(f"  Dim: {features.shape[1]} -> {X_ica.shape[1]}")
    mi=compute_feature_importance(features,labels); top5=np.argsort(mi)[-5:][::-1]
    print(f"  Top-5 MI features: {list(top5)}")

    # 3. Graph construction (for architectural demonstration)
    print("\n[3] Building dynamic graphs...")
    graphs=build_static_graphs(data,time_window=43200)
    print(f"  {len(graphs)} graph snapshots (12h windows)")

    # 4. Train lightweight GNN on one snapshot
    print("\n[4] Training DIDS-MFL GNN (1 snapshot)...")
    gnn=DIDS_MFL(in_dim=40,hidden_dim=64,latent_dim=32,num_classes=2,num_factors=4)
    g=graphs[0]
    x=torch.zeros(g.num_nodes,40)
    for i in range(g.num_nodes):
        mask=(g.edge_index[0]==i)|(g.edge_index[1]==i)
        if mask.sum()>0: x[i]=g.edge_attr[mask].mean(dim=0)
    opt=torch.optim.Adam(gnn.parameters(),lr=0.001)
    gnn.train()
    for ep in range(30):
        opt.zero_grad()
        logits,z=gnn(x,g.edge_index,return_latent=True)
        nl=torch.zeros(g.num_nodes,dtype=torch.long); nc=torch.zeros(g.num_nodes,dtype=torch.long)
        for j in range(g.edge_index.shape[1]):
            u,v=g.edge_index[0,j].item(),g.edge_index[1,j].item()
            nl[u]+=g.edge_label[j].item(); nl[v]+=g.edge_label[j].item()
            nc[u]+=1; nc[v]+=1
        mv=nc>0; nl[mv]=(nl[mv].float()/nc[mv].float()>0.5).long()
        loss=gnn.compute_loss(logits[mv],nl[mv],z[mv])
        loss.backward(); opt.step()
    gnn.eval()
    with torch.no_grad():
        _,z_gnn=gnn(x,g.edge_index,return_latent=True)
    print(f"  GNN trained: latent dim={z_gnn.shape[1]}, loss={loss.item():.4f}")

    # 5. Edge classifier with disentangled features
    print("\n[5] Edge classifier on disentangled features...")
    Xt=torch.tensor(X_ica,dtype=torch.float32); yt=torch.tensor(labels,dtype=torch.long)
    Xtr,Xte,ytr,yte=train_test_split(Xt,yt,test_size=0.3,random_state=SEED,stratify=yt.numpy())
    
    clf=EdgeClassifier(in_dim=X_ica.shape[1],hidden_dim=128,num_classes=2)
    opt=torch.optim.Adam(clf.parameters(),lr=0.001)
    for ep in range(30):
        clf.train(); opt.zero_grad()
        loss=F.cross_entropy(clf(Xtr),ytr); loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad(): dpreds=clf(Xte).argmax(dim=1).numpy()
    
    bin_res={'DIDS-MFL':{
        'accuracy':float(accuracy_score(yte.numpy(),dpreds)),
        'f1':float(f1_score(yte.numpy(),dpreds,average='binary')),
        'precision':float(precision_score(yte.numpy(),dpreds,average='binary')),
        'recall':float(recall_score(yte.numpy(),dpreds,average='binary'))}}

    # 6. Baselines
    print("\n[6] Baselines...")
    rf=RandomForestClassifier(n_estimators=100,max_depth=15,random_state=SEED,n_jobs=-1)
    rf.fit(Xtr.numpy(),ytr.numpy()); rfp=rf.predict(Xte.numpy())
    bin_res['Random Forest']={'accuracy':float(accuracy_score(yte.numpy(),rfp)),'f1':float(f1_score(yte.numpy(),rfp,average='binary')),'precision':float(precision_score(yte.numpy(),rfp,average='binary')),'recall':float(recall_score(yte.numpy(),rfp,average='binary'))}
    
    nsvm=min(10000,len(Xtr)); isvm=np.random.choice(len(Xtr),nsvm,replace=False)
    svm=SVC(kernel='rbf',random_state=SEED)
    svm.fit(Xtr.numpy()[isvm],ytr.numpy()[isvm]); sp=svm.predict(Xte.numpy())
    bin_res['SVM']={'accuracy':float(accuracy_score(yte.numpy(),sp)),'f1':float(f1_score(yte.numpy(),sp,average='binary')),'precision':float(precision_score(yte.numpy(),sp,average='binary')),'recall':float(recall_score(yte.numpy(),sp,average='binary'))}
    
    for n,r in bin_res.items(): print(f"  {n}: Acc={r['accuracy']:.4f}, F1={r['f1']:.4f}")

    # 7. Multi-class
    print("\n[7] Multi-class classification...")
    le=LabelEncoder(); ymc=le.fit_transform(attacks)
    Xtrm,Xtem,ytrm,ytem=train_test_split(X_ica,ymc,test_size=0.3,random_state=SEED,stratify=ymc)
    rfm=RandomForestClassifier(n_estimators=100,max_depth=15,random_state=SEED,n_jobs=-1)
    rfm.fit(Xtrm,ytrm); rmp=rfm.predict(Xtem)
    
    pcf1={}
    for i in range(len(le.classes_)):
        tp=((rmp==i)&(ytem==i)).sum(); fp=((rmp==i)&(ytem!=i)).sum(); fn=((rmp!=i)&(ytem==i)).sum()
        prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
        pcf1[f'class_{le.classes_[i]}']=float(2*prec*rec/max(prec+rec,1e-10))
        print(f"  Class {le.classes_[i]}: F1={pcf1[f'class_{le.classes_[i]}']:.4f}")
    
    mc_res={'accuracy':float(accuracy_score(ytem,rmp)),'macro_f1':float(f1_score(ytem,rmp,average='macro')),'weighted_f1':float(f1_score(ytem,rmp,average='weighted')),'per_class_f1':pcf1,'class_names':[f'Class_{c}' for c in le.classes_],'confusion_matrix':confusion_matrix(ytem,rmp).tolist()}

    # 8. Few-shot
    print("\n[8] Few-shot evaluation...")
    eps=create_few_shot_episodes(X_ica,attacks,n_way=5,n_shot=5,n_query=15,n_episodes=100)
    if eps:
        pn=PrototypeNetwork(in_dim=X_ica.shape[1],hidden_dim=128)
        accs=[]
        for ep in eps:
            sfe=pn(ep['support_feats'])
            qfe=pn(ep['query_feats'])
            pr=pn.compute_prototypes(sfe,ep['support_labels'],ep['n_classes'])
            preds=pn.predict(qfe,pr).numpy()
            accs.append(accuracy_score(ep['query_labels'].numpy(),preds))
        fs_res={'mean_accuracy':float(np.mean(accs)),'std_accuracy':float(np.std(accs)),'n_episodes':len(eps)}
        print(f"  Mean Acc: {fs_res['mean_accuracy']:.4f} +/- {fs_res['std_accuracy']:.4f}")
    else: fs_res={'error':'No episodes'}

    # 9. Unknown attack
    print("\n[9] Unknown attack detection...")
    acs=sorted(set(attacks)-{2}); uapc=[]
    for ho in acs:
        kn=[c for c in acs if c!=ho]+[2]; un=[ho]
        r=evaluate_open_set(X_ica,attacks,kn,un); r['held_out_class']=int(ho); uapc.append(r)
        print(f"  Hold-out {ho}: F1={r['f1']:.4f}")
    uavg={k:float(np.mean([x[k] for x in uapc])) for k in ['accuracy','f1','precision','recall']}
    uavg['per_class']=uapc
    print(f"  Average F1: {uavg['f1']:.4f}")

    # Save
    all_res={
        'binary_classification':bin_res,'multiclass_classification':mc_res,
        'few_shot':fs_res,'unknown_attack':uavg,
        'feature_importance':mi.tolist(),'top5_features':top5.tolist(),
        'data_statistics':{'total_flows':int(len(features)),'benign':int((labels==0).sum()),'attack':int((labels==1).sum()),'n_attack_classes':int(len(set(attacks))),'feature_dim':int(features.shape[1])}}
    os.makedirs('../outputs',exist_ok=True)
    with open('outputs/all_results.json','w') as f: json.dump(all_res,f,indent=2)
    print(f"\n{'='*60}\nSaved to outputs/all_results.json")
    return all_res

if __name__=='__main__': main()
