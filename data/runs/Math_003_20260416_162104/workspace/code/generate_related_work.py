import json

related_work = {
  "paper_000": "Attention Is All You Need. Introduces the Transformer architecture, foundational for modern LMs.",
  "paper_001": "Generative Language Modeling for Automated Theorem Proving (GPT-f). Explores transformer-based LMs for automated theorem proving in Metamath.",
  "paper_003": "Mastering the game of Go with deep neural networks and tree search (AlphaGo). Demonstrates the power of combining neural networks with search algorithms (MCTS) for complex problem-solving."
}

with open('outputs/related_work_contract.json', 'w') as f:
    json.dump(related_work, f, indent=2)

