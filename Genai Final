import os
from getpass import getpass
# import json
# import sys

# --- 1. Dependency Installation (from h9mvOsLO1BPi)
# !pip install -q langchain langchain-core langchain-community langchain-groq langchain-text-splitters
# !pip install -q faiss-cpu sentence-transformers
print('✅ All packages installed!')

# --- 2. API Key Setup (from 4ySKDlfN1BPl)
os.environ['GROQ_API_KEY'] = getpass('🔑 Enter Groq API Key: ')
print('✅ API Key set!')

# --- 3. All Necessary Imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 4. LLM Initialization (from LFEcHlDJ1BPm)
llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0.7, max_tokens=1024)

response = llm.invoke('Say "LangChain 0.3 ready!" in one line.')
print('✅ LLM Test:', response.content)


# --- 5. Q&A Chatbot with Memory (from 8gqWJUD61BPm)
print('\n' + '=' * 55)
print('   🤖 MINI PROJECT — Q&A Chatbot')
print('=' * 55)

chat_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful AI assistant. Be clear and concise.'),
    MessagesPlaceholder(variable_name='history'),
    ('human', '{input}'),
])

chat_chain = chat_prompt | llm | StrOutputParser()

chat_history = []

def chat(user_input):
    response = chat_chain.invoke({'input': user_input, 'history': chat_history})
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    return response

# --- 6. Feature 1: Text Summarizer (from xQ13njA91BPn)
print('\n' + '=' * 55)
print('   📄 Feature 1: Summarizer')
print('=' * 55)

summarize_prompt = PromptTemplate(
    input_variables=['text'],
    template="""
Summarize the following text:

{text}

Provide:
1. TL;DR (1 sentence)
2. Key Points (3-5 bullets)
3. Main Takeaway
"""
)

summarize_chain = summarize_prompt | llm | StrOutputParser()

def summarize(text):
    return summarize_chain.invoke({'text': text})

# --- 7. Feature 2: Translator (from _YG339Gw1BPo)
print('\n' + '=' * 55)
print('   🌐 Feature 2: Translator')
print('=' * 55)

translate_prompt = PromptTemplate(
    input_variables=['text', 'target_language'],
    template="""
Translate to {target_language}:

ORIGINAL: {text}

Provide:
- TRANSLATION:
- PRONUNCIATION: (if non-latin)
- CULTURAL NOTE: (if relevant)
"""
)

translate_chain = translate_prompt | llm | StrOutputParser()
def translate(text, lang): return translate_chain.invoke({'text': text, 'target_language': lang})

# --- 8. Feature 3: Code Generator (from gND5b2Xa1BPp)
print('\n' + '=' * 55)
print('   ⚡ Feature 3: Code Generator')
print('=' * 55)

code_prompt = PromptTemplate(
    input_variables=['description', 'language'],
    template="Write clean, commented {language} code for: {description}\n\nCODE:"
)
explain_prompt = PromptTemplate(
    input_variables=['code'],
    template="Explain this code simply in 3-5 bullet points:\n{code}\n\nEXPLANATION:"
)

code_chain    = code_prompt    | llm | StrOutputParser()
explain_chain = explain_prompt | llm | StrOutputParser()

def generate_code(description, language='Python'):
    code = code_chain.invoke({'description': description, 'language': language})
    explanation = explain_chain.invoke({'code': code})
    return {'code': code, 'explanation': explanation}

# --- 9. Feature 4: Quiz Generator (from 4KzYpiuu1BPp)
print('\n' + '=' * 55)
print('   🧠 Feature 4: Quiz Generator')
print('=' * 55)

quiz_prompt = PromptTemplate(
    input_variables=['topic', 'num_questions'],
    template="""
Create {num_questions} multiple choice quiz questions about: {topic}
Make them progressively harder.

Return ONLY valid JSON in this exact format (no extra text):
{{
  "topic": "topic name",
  "questions": [
    {{
      "question": "...",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer": "A",
      "explanation": "..."
    }}
  ]
}}
"""
)

quiz_chain = quiz_prompt | llm | JsonOutputParser()

def generate_quiz(topic, num_questions=3):
    try:
        return quiz_chain.invoke({'topic': topic, 'num_questions': num_questions})
    except Exception:
        raw = (quiz_prompt | llm | StrOutputParser()).invoke({'topic': topic, 'num_questions': num_questions})
        return {'raw': raw}



def display_quiz(data):
    if 'raw' in data: print(data['raw']); return
    print(f"📚 Topic: {data.get('topic','')}\n")
    for i, q in enumerate(data.get('questions', []), 1):
        q = q if isinstance(q, dict) else q.__dict__
        print(f"Q{i}: {q['question']}")
        for opt in q['options']: print(f'   {opt}')
        print(f"✅ {q['answer']}  💡 {q['explanation']}\n")

# --- 10. Feature 5: RAG System (from y67BHPhW1BPq)
print('\n' + '=' * 55)
print('   🔍 Feature 5: RAG Document Q&A')
print('=' * 55)

print('Loading embeddings...')
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

docs = [
    "LangChain was created by Harrison Chase in 2022. It supports Python and JavaScript.",
    "RAG (Retrieval-Augmented Generation) combines document retrieval with text generation.",
    "FAISS is Facebook's vector store. Alternatives: Pinecone, Chroma, Weaviate, Qdrant.",
    "LCEL uses the pipe operator | to chain: prompt | llm | parser. Replaces LLMChain.",
    "LangSmith debugs/monitors LLM apps. LangServe deploys chains as APIs.",
    "LangChain Agents use ReAct framework to reason and call tools like search or calculator."
]

splits = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30).split_documents(
    [Document(page_content=t) for t in docs]
)
retriever = FAISS.from_documents(splits, embeddings).as_retriever(search_kwargs={'k': 3})
print(f'✅ Vector store ready! ({len(splits)} chunks)')

rag_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer using ONLY the context below:\n\n{context}'),
    ('human', '{question}')
])

rag_chain = (
    {'context': retriever | (lambda docs: '\n\n'.join(d.page_content for d in docs)),
     'question': RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)

def ask_rag(q):
    print(f'❓ {q}')
    print(f'💬 {rag_chain.invoke(q)}\n')

# --- 11. Interactive Demo (from sVjJUZVx1BPr)
chat_history.clear() # Clear chat history for a fresh run

def run_demo():
    while True:
        print("MENU:")
        print("1. Chat")
        print("2. Summarize")
        print("3. Translate")
        print("4. Code Gen")
        print("5. Quiz")
        print("6. RAG Q&A")
        print("0. Exit")

        c = input("Choose (0-6): ").strip()

        if c == '0':
            print('\n👋 Goodbye! Cell execution stopped.')
            return   # exits the function → cell stops cleanly

        elif c == '1':
            print('💬 Chat mode — type "back" to return to menu')
            while True:
                u = input('You: ').strip()
                if u.lower() == 'back':
                    break
                if u:
                    print('Bot:', chat(u))

        elif c == '2':
            print('📄 Paste text. Type END on a new line when done:')
            lines = []
            while True:
                l = input()
                if l.strip().upper() == 'END':
                    break
                lines.append(l)
            if lines:
                print('\n📋 SUMMARY:\n', summarize('\n'.join(lines)))

        elif c == '3':
            t = input('🌐 Text to translate: ').strip()
            l = input('Target language: ').strip()
            if t and l:
                print(translate(t, l))

        elif c == '4':
            d = input('⚡ Describe the code task: ').strip()
            l = input('Language (default Python): ').strip() or 'Python'
            if d:
                print('\n⏳ Generating...')
                r = generate_code(d, l)
                print('\n📦 CODE:\n', r['code'])
                print('\n📖 EXPLANATION:\n', r['explanation'])

        elif c == '5':
            t = input('🧠 Quiz topic: ').strip()
            n = input('Number of questions (default 3): ').strip()
            n = int(n) if n.isdigit() else 3
            if t:
                print('\n⏳ Generating quiz...')
                display_quiz(generate_quiz(t, n))

        elif c == '6':
            q = input('🔍 Ask the knowledge base: ').strip()
            if q:
                ask_rag(q)

        else:
            print('❌ Invalid choice. Please enter 0–6.')

# Run the demo
run_demo()
