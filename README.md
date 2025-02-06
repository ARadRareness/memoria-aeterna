# Memoria Aeterna

Memoria Aeterna is a memory system designed for large language models, enabling them to maintain persistent memory across conversations. By integrating with my othe project AMP, it provides a robust solution for maintaining context and building long-term knowledge bases for AI interactions.

## Getting Started

### 1. Prerequisites
First, install AMP by following the instructions at:
https://github.com/ARadRareness/AMP.git

### 2. Clone this repository
```bash
git clone https://github.com/yourusername/memoria-aeterna.git
cd memoria-aeterna
```

### 3. (Optional) Create and activate a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

### 4. Install requirements
```bash
pip install -r requirements.txt
```

## Running

### 1. Start the memory server
```bash
python memory_server.py
```

### 2. Start the chat client
```bash
python start_chat.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.