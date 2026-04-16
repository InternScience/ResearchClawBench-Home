import json
import os

os.makedirs('outputs', exist_ok=True)

contract = {
    "related_work_extraction": [
        {
            "paper": "paper_000.pdf",
            "title": "Global glacier change in the 21st century: Every increase in temperature matters",
            "relevance": "Provides context on the importance of global glacier mass loss for sea level rise, water resources, and natural hazards."
        },
        {
            "paper": "paper_001.pdf",
            "title": "Partitioning the Uncertainty of Ensemble Projections of Global Glacier Mass Change",
            "relevance": "Highlights the necessity of understanding and quantifying uncertainties in glacier mass change projections."
        },
        {
            "paper": "paper_002.pdf",
            "title": "Global glacier mass changes and their contributions to sea-level rise from 1961 to 2016",
            "relevance": "Establishes a historical baseline for global glacier mass loss and its contribution to sea-level rise."
        },
        {
            "paper": "paper_003.pdf",
            "title": "GlacierMIP –A model intercomparison of global-scale glacier mass-balance models and projections",
            "relevance": "Demonstrates the value of intercomparison exercises for improving confidence in global glacier mass change estimates."
        },
        {
            "paper": "paper_004.pdf",
            "title": "Accelerated global glacier mass loss in the early twenty-first century",
            "relevance": "Confirms the acceleration of global glacier mass loss in the early 21st century, aligning with our decadal trend findings."
        }
    ]
}

with open('outputs/related_work_contract.json', 'w') as f:
    json.dump(contract, f, indent=4)

print("Saved outputs/related_work_contract.json")
