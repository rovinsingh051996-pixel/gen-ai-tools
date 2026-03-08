# 🤖 LangChain AI Multi-Feature Demo

An interactive Python application built with **LangChain 0.3** and **Groq API** that demonstrates modern AI capabilities through 6 powerful features.

## ✨ Features

### 1. 💬 Q&A Chatbot with Memory
- Conversational AI assistant with chat history
- Maintains context across multiple exchanges
- Clear and concise responses

### 2. 📄 Text Summarizer
- Generates concise summaries in three formats:
  - **TL;DR**: 1-sentence summary
  - **Key Points**: 3-5 bullet points
  - **Main Takeaway**: Core insight

### 3. 🌐 Translator
- Translate text to any language
- Provides pronunciation guides
- Includes cultural notes when relevant

### 4. ⚡ Code Generator
- Generates clean, well-commented code
- Supports multiple programming languages
- Auto-generates code explanations

### 5. 🧠 Quiz Generator
- Creates multiple-choice quizzes on any topic
- Progressively increasing difficulty
- Includes explanations for answers
- Returns structured JSON format

### 6. 🔍 RAG Document Q&A
- **Retrieval-Augmented Generation** system
- Vector database with FAISS
- Ask questions about a knowledge base
- Accurate, context-aware responses

## 🛠️ Tech Stack

- **LangChain 0.3**: LLM framework with LCEL (LangChain Expression Language)
- **Groq API**: Fast LLM inference (llama-3.1-8b-instant)
- **FAISS**: Vector database for semantic search
- **HuggingFace Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Python 3.8+**

## 📦 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd gen-ai

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
langchain==0.3.*
langchain-core
langchain-community
langchain-groq
langchain-text-splitters
faiss-cpu
sentence-transformers
```

## 🚀 Quick Start

1. **Get a Groq API Key**
   - Sign up at [console.groq.com](https://console.groq.com)
   - Create an API key in your account settings

2. **Run the application**
   ```bash
   python gen\ ai\ final.py
   ```

3. **Enter your Groq API Key** when prompted

4. **Choose a feature** from the interactive menu

## 📋 Usage Examples

### Chat Mode
```
Choose (0-6): 1
💬 Chat mode — type "back" to return to menu
You: What is machine learning?
Bot: Machine learning is a subset of AI...
```

### Summarize
```
Choose (0-6): 2
📄 Paste text. Type END on a new line when done:
[Paste long text]
END

📋 SUMMARY:
- TL;DR: [1 sentence summary]
- Key Points: [Bullet points]
- Main Takeaway: [Insight]
```

### Translate
```
Choose (0-6): 3
🌐 Text to translate: Hello, how are you?
Target language: Spanish

TRANSLATION: Hola, ¿cómo estás?
PRONUNCIATION: OH-lah, KOH-moh ess-TAHS
CULTURAL NOTE: ...
```

### Generate Code
```
Choose (0-6): 4
⚡ Describe the code task: Function to reverse a string
Language (default Python): Python

📦 CODE:
def reverse_string(s):
    return s[::-1]

📖 EXPLANATION:
- Uses Python's slice notation
- Efficient O(n) time complexity
- Works with any string
```

### Quiz Generator
```
Choose (0-6): 5
🧠 Quiz topic: Python basics
Number of questions (default 3): 3

📚 Topic: Python basics

Q1: What is a list in Python?
   A) A dictionary
   B) An ordered, mutable sequence
   C) A string
   D) A set

✅ B)  💡 Lists are fundamental data structures...
```

### RAG Q&A
```
Choose (0-6): 6
🔍 Ask the knowledge base: What is LCEL?

❓ What is LCEL?
💬 LCEL (LangChain Expression Language) uses the pipe operator...
```

## 🏗️ Architecture

### LCEL Chain Pattern
```python
chain = prompt | llm | output_parser
```

This modern approach replaces traditional `LLMChain`, providing:
- Clean, readable syntax
- Automatic streaming support
- Better debugging

### RAG System
```
User Query
    ↓
Embedding Generator
    ↓
FAISS Vector Store Retrieval
    ↓
Context + Prompt
    ↓
LLM Response
```

## 🔑 Key Components

| Component | Purpose |
|-----------|---------|
| `ChatGroq` | LLM interface to Groq API |
| `ChatPromptTemplate` | Structure system + context + user input |
| `MessagesPlaceholder` | Insert chat history dynamically |
| `HuggingFaceEmbeddings` | Convert text to vectors |
| `FAISS` | Fast vector similarity search |
| `JsonOutputParser` | Extract structured data from LLM |
| `RunnablePassthrough` | Pass data through pipeline unchanged |

## 🔧 Configuration

### Model Settings
```python
llm = ChatGroq(
    model='llama-3.1-8b-instant',
    temperature=0.7,  # Creativity (0=strict, 1=creative)
    max_tokens=1024   # Response length
)
```

### Vector Store
```python
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
retriever = FAISS.from_documents(
    splits, 
    embeddings,
    search_kwargs={'k': 3}  # Return top 3 relevant docs
)
```

## 📚 Learning Resources

- [LangChain Documentation](https://python.langchain.com)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/concepts/lcel)
- [Groq API Docs](https://console.groq.com/docs)
- [RAG Concepts](https://python.langchain.com/docs/use_cases/qa_structured_sources)
- [FAISS Tutorial](https://github.com/facebookresearch/faiss/wiki)

## 🎯 Use Cases

- **Education**: Learn LLM concepts through interactive demos
- **Prototyping**: Quick test of LangChain capabilities
- **Content Creation**: Generate summaries, translations, and code
- **Knowledge Management**: Build RAG systems for document Q&A
- **Multi-task AI**: Single interface for diverse AI features

## ⚙️ Environment Variables

```bash
# Required
GROQ_API_KEY=your_api_key_here

# Optional (for production)
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=project_name
```

## 🐛 Troubleshooting

### Package Import Errors
```bash
pip install --upgrade langchain langchain-core langchain-community
```

### Groq API Connection Issues
- Verify API key is correct
- Check internet connection
- Ensure API key has active credits

### FAISS Build Errors (Windows)
```bash
pip install faiss-cpu  # Already included in requirements
```

### Out of Memory
- Reduce chunk_size in RecursiveCharacterTextSplitter
- Use fewer documents in RAG
- Reduce max_tokens

## 📄 License

MIT License - Free to use, modify, and distribute

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 💡 Ideas for Enhancement

- [ ] Add WebUI with Streamlit/FastAPI
- [ ] Implement multi-document RAG
- [ ] Add conversation persistence (database)
- [ ] Support for custom knowledge bases
- [ ] API endpoint deployment with LangServe
- [ ] Multi-language support
- [ ] Performance benchmarking

## 📞 Support

For issues, questions, or feedback:
- Open a GitHub Issue
- Check existing discussions
- Review the documentation

---

**Built with ❤️ using LangChain 0.3 and Groq API**

Happy coding! 🚀
