from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document 

import sys
import json
import os
import logging
from getpass import getpass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API Key
if not os.getenv('GROQ_API_KEY'):
    api_key = getpass('🔑 Enter Groq API Key: ').strip()
    if not api_key:
        logger.error('❌ API Key is required!')
        sys.exit(1)
    os.environ['GROQ_API_KEY'] = api_key
else:
    logger.info('✅ API Key loaded from environment')

try:
    llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0.7, max_tokens=1024, timeout=30)
    response = llm.invoke('Say "LangChain 0.3 ready!" in one line.')
    logger.info(f'✅ LLM Test: {response.content}')
except Exception as e:
    logger.error(f'❌ LLM initialization failed: {e}')
    sys.exit(1)

# --- 1. Chat with Memory
chat_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful AI assistant. Be clear and concise.'),
    MessagesPlaceholder(variable_name='history'),
    ('human', '{input}'),
])

chain = chat_prompt | llm | StrOutputParser()
chat_history = []

def chat(user_input):
    if not user_input or not user_input.strip():
        logger.warning('⚠️  Empty chat input')
        return 'Please enter a valid message.'
    try:
        response = chain.invoke({'input': user_input.strip(), 'history': chat_history})
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        return response
    except Exception as e:
        logger.error(f'Chat error: {e}')
        return f'❌ Error: {str(e)[:100]}'

# --- 2. Text Summarizer
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
    if not text or len(text.strip()) < 10:
        logger.warning('⚠️  Text too short for summarization')
        return 'Please provide at least 10 characters of text.'
    try:
        result = summarize_chain.invoke({'text': text.strip()})
        logger.info('✅ Text summarized')
        return result
    except Exception as e:
        logger.error(f'Summarize error: {e}')
        return f'❌ Error: {str(e)[:100]}'

# --- 3. Translator
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

def translate(text, lang):
    if not text or not text.strip():
        logger.warning('⚠️  Empty text for translation')
        return 'Please provide text to translate.'
    if not lang or not lang.strip():
        logger.warning('⚠️  No language specified')
        return 'Please specify a target language.'
    try:
        result = translate_chain.invoke({'text': text.strip(), 'target_language': lang.strip()})
        logger.info(f'✅ Translated to {lang}')
        return result
    except Exception as e:
        logger.error(f'Translate error: {e}')
        return f'❌ Error: {str(e)[:100]}'

# --- 4. Code Generator
code_prompt = PromptTemplate(
    input_variables=['description', 'language'],
    template="Write clean, commented {language} code for: {description}\n\nCODE:"
)
explain_prompt = PromptTemplate(
    input_variables=['code'],
    template="Explain this code simply in 3-5 bullet points:\n{code}\n\nEXPLANATION:"
)

code_chain = code_prompt | llm | StrOutputParser()
explain_chain = explain_prompt | llm | StrOutputParser()

def generate_code(description, language='Python'):
    if not description or len(description.strip()) < 5:
        logger.warning('⚠️  Description too short')
        return {'code': 'Error: Please provide a longer description.', 'explanation': ''}
    try:
        code = code_chain.invoke({'description': description.strip(), 'language': language.strip()})
        explanation = explain_chain.invoke({'code': code})
        logger.info(f'✅ Generated {language} code')
        return {'code': code, 'explanation': explanation}
    except Exception as e:
        logger.error(f'Code generation error: {e}')
        return {'code': f'❌ Error: {str(e)[:100]}', 'explanation': 'Failed to generate code'}

# --- 5. Quiz Generator
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
    if not topic or len(topic.strip()) < 3:
        logger.warning('⚠️  Topic too short')
        return {'error': 'Please provide a topic of at least 3 characters.'}
    if not isinstance(num_questions, int) or num_questions < 1 or num_questions > 10:
        logger.warning('⚠️  Invalid number of questions')
        num_questions = 3
    try:
        result = quiz_chain.invoke({'topic': topic.strip(), 'num_questions': num_questions})
        logger.info(f'✅ Generated {num_questions} quiz questions')
        return result
    except Exception as e:
        logger.warning(f'JSON parsing error, using fallback: {e}')
        try:
            raw = (quiz_prompt | llm | StrOutputParser()).invoke({'topic': topic.strip(), 'num_questions': num_questions})
            return {'raw': raw}
        except Exception as e2:
            logger.error(f'Quiz generation error: {e2}')
            return {'error': f'Failed to generate quiz: {str(e2)[:100]}'}

def display_quiz(data):
    try:
        if 'error' in data:
            print(f"❌ {data['error']}")
            return
        if 'raw' in data:
            print(data['raw'])
            return
        print(f"📚 Topic: {data.get('topic','')}\n")
        for i, q in enumerate(data.get('questions', []), 1):
            q = q if isinstance(q, dict) else q.__dict__
            print(f"Q{i}: {q['question']}")
            for opt in q['options']: print(f'   {opt}')
            print(f"✅ {q['answer']}  💡 {q['explanation']}\n")
    except Exception as e:
        logger.error(f'Display quiz error: {e}')
        print(f'❌ Error displaying quiz: {str(e)[:100]}')

# --- 6. RAG System
try:
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
    logger.info(f'✅ Vector store ready! ({len(splits)} chunks)')
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
except Exception as e:
    logger.error(f'RAG system initialization failed: {e}')
    rag_chain = None

def ask_rag(q):
    if not rag_chain:
        print('❌ RAG system not available. Please check the logs.')
        return
    if not q or not q.strip():
        logger.warning('⚠️  Empty RAG query')
        print('Please ask a valid question.')
        return
    try:
        print(f'❓ {q}')
        result = rag_chain.invoke(q.strip())
        print(f'💬 {result}\n')
        logger.info('✅ RAG query processed')
    except Exception as e:
        logger.error(f'RAG query error: {e}')
        print(f'❌ Error: {str(e)[:100]}')

def run_demo():
    while True:
        try:
            print("\n" + "="*55)
            print("MENU:")
            print("1. Chat")
            print("2. Summarize")
            print("3. Translate")
            print("4. Code Gen")
            print("5. Quiz")
            print("6. RAG Q&A")
            print("0. Exit")
            print("="*55)

            c = input("Choose (0-6): ").strip()

            if c == '0':
                print('\n👋 Goodbye!')
                logger.info('Application ended by user')
                return

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
                    try:
                        l = input()
                        if l.strip().upper() == 'END':
                            break
                        lines.append(l)
                    except KeyboardInterrupt:
                        print('\n⚠️  Operation cancelled.')
                        break
                if lines:
                    print('\n📋 SUMMARY:\n', summarize('\n'.join(lines)))

            elif c == '3':
                t = input('🌐 Text to translate: ').strip()
                l = input('Target language: ').strip()
                if t and l:
                    print('⏳ Translating...')
                    print(translate(t, l))
                else:
                    print('❌ Both text and language are required.')

            elif c == '4':
                d = input('⚡ Describe the code task: ').strip()
                l = input('Language (default Python): ').strip() or 'Python'
                if d:
                    print('\n⏳ Generating...')
                    r = generate_code(d, l)
                    print('\n📦 CODE:\n', r['code'])
                    print('\n📖 EXPLANATION:\n', r['explanation'])
                else:
                    print('❌ Please provide a description.')

            elif c == '5':
                t = input('🧠 Quiz topic: ').strip()
                n = input('Number of questions (default 3): ').strip()
                try:
                    n = int(n) if n.isdigit() else 3
                    n = max(1, min(n, 10))  # Clamp between 1-10
                except ValueError:
                    n = 3
                if t:
                    print('\n⏳ Generating quiz...')
                    display_quiz(generate_quiz(t, n))
                else:
                    print('❌ Please provide a topic.')

            elif c == '6':
                q = input('🔍 Ask the knowledge base: ').strip()
                if q:
                    ask_rag(q)
                else:
                    print('❌ Please ask a valid question.')

            else:
                print('❌ Invalid choice. Please enter 0–6.')
        except KeyboardInterrupt:
            print('\n\n👋 Application interrupted by user.')
            logger.info('Application interrupted')
            break
        except Exception as e:
            logger.error(f'Unexpected error in menu: {e}')
            print(f'❌ Unexpected error: {str(e)[:100]}')

# Run the demo
if __name__ == '__main__':
    try:
        logger.info('Application started')
        run_demo()
    except Exception as e:
        logger.critical(f'Fatal error: {e}')
        print(f'❌ Fatal error: {e}')
    finally:
        logger.info('Application ended')
        print('\n✅ Thank you for using AI Studio!')
