"""Split the two input PDB complexes into per-chain PDB files and a
protein-only version, used by Foldseek and per-chain US-align comparisons.
"""
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
import warnings

warnings.filterwarnings("ignore")

PROT3 = {
    "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE",
}

WORK = Path(__file__).resolve().parents[1]


class ChainSel(Select):
    def __init__(self, cid):
        self.cid = cid

    def accept_chain(self, ch):
        return ch.id == self.cid

    def accept_residue(self, r):
        return r.id[0] == " "


class ProtSel(Select):
    def accept_residue(self, r):
        return r.id[0] == " " and r.resname.strip() in PROT3


def main():
    parser = PDBParser(QUIET=True)
    io = PDBIO()
    out_chains = WORK / "outputs" / "chains"
    out_chains.mkdir(parents=True, exist_ok=True)

    for fp_rel, name in [("data/7xg4.pdb", "7xg4"), ("data/6n40.pdb", "6n40")]:
        struct = parser.get_structure(name, WORK / fp_rel)
        # whole protein-only structure
        io.set_structure(struct)
        io.save(str(WORK / "outputs" / f"{name}_prot.pdb"), ProtSel())
        # individual chains (all chain types)
        for chain in struct[0]:
            io.set_structure(struct)
            io.save(str(out_chains / f"{name}_{chain.id}.pdb"), ChainSel(chain.id))


if __name__ == "__main__":
    main()
