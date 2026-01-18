import os
from typing import List, Dict

from qdrant_setup import QdrantSetup
from embedding_generator import HindiEmbeddingGenerator, get_embedding_function
from llm_manager import LLMManager, get_llm, get_llm_with_provider
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import OpenAI if available for better fallback
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class HindiRAGSystem:
    def __init__(self, llm_provider=None, model_kwargs=None):
        """
        Initialize the Hindi RAG system
        """
        # Setup Qdrant client
        qdrant_setup = QdrantSetup()
        self.qdrant_client = qdrant_setup.get_client()
        self.collection_name = qdrant_setup.get_collection_name()

        # Setup embedding generator
        self.embedding_generator = HindiEmbeddingGenerator()

        # Setup LLM with provider and model kwargs
        if llm_provider or model_kwargs:
            self.llm = get_llm_with_provider(provider=llm_provider, model_kwargs=model_kwargs)
        else:
            self.llm = get_llm()
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents from Qdrant based on the query
        """
        # Generate embedding for the query using the same embedding function as during ingestion
        query_embedding = self.embedding_generator.get_embedding(query)

        # Use the query_points method which should be available in current version
        from qdrant_client.http import models

        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k
        )

        # Extract relevant documents
        retrieved_docs = []
        for result in search_result.points:
            # Handle case where payload is None
            payload = result.payload if result.payload is not None else {}

            doc = {
                'score': result.score or 0,
                'title': payload.get('title', '') if payload else '',
                'author': payload.get('author', '') if payload else '',
                'genre': payload.get('genre', '') if payload else '',
                'text': payload.get('full_text', '') if payload else '',
                'source_file': payload.get('source_file', '') if payload else ''
            }
            retrieved_docs.append(doc)

        return retrieved_docs
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate an answer based on the query and retrieved documents using configurable LLM
        """
        # Check if the query is asking for poem generation
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["कविता लिखो", "कविता बनाओ", "गीत लिखो", "गीत बनाओ", "poem", "गाना", "गीत", "कविता लिखें", "लिखो कविता", "बनाओ कविता", "एक कविता लिखो"]):
            return self.generate_poem(query, context_docs)

        # Limit the number of documents to prevent context overflow
        # Take only the top 2 most relevant documents
        limited_docs = context_docs[:2] if len(context_docs) > 2 else context_docs

        # Format context from retrieved documents with metadata, ensuring very short length
        formatted_contexts = []
        for i, doc in enumerate(limited_docs, 1):
            # Limit text length significantly to prevent overflow
            text_snippet = doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
            formatted_context = f"[{i}] {doc['title']} ({doc['author']}): {text_snippet}"
            formatted_contexts.append(formatted_context)

        context_str = " | ".join(formatted_contexts)

        # Construct System Prompt
        system_prompt_text = """
        You are a helpful assistant for Hindi literature and poetry.
        Use the provided context to answer the user's question accurately in Hindi.
        If the answer is not in the context, say "I don't know based on the provided documents."
        Do not hallucinate.
        """

        # Construct User Prompt
        user_prompt = f"""
        Context:
        {context_str}

        Question: {query}

        Answer (in Hindi):
        """

        # Create a much shorter prompt template to fit within model limits
        # Reformatted for better instruction following by various models
        template = user_prompt

        try:
            prompt = PromptTemplate.from_template(template)

            # Create the chain
            chain = (
                {"context": lambda x: x["context"], "question": lambda x: x["question"]}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Generate response using the configured LLM
            response = chain.invoke({"context": context, "question": query})

            # Check if response is meaningful - it should not just be the prompt template
            if response and len(response.strip()) > 20 and not response.strip().startswith("Context:") and not response.strip().startswith("Question:"):
                return response
            else:
                print(f"LLM returned empty, minimal, or template response: '{response[:50]}...'. Falling back to document summary and synthesis.")
        except Exception as e:
            print(f"Local LLM generation failed: {e}")

        # If we reach here, either the LLM failed with an exception or returned an empty response
        # Try OpenAI as fallback if available
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                print("Trying OpenAI as fallback...")
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                # Format a simpler prompt for OpenAI
                simplified_context = ""
                for doc in limited_docs:
                    simplified_context += f"Title: {doc['title']} by {doc['author']}\n"
                    simplified_context += f"Content: {doc['text'][:300]}...\n\n"

                full_prompt = f"""
                Context information is below.
                ----------------
                {simplified_context}
                ----------------
                Given the context information and not prior knowledge, answer the query in Hindi if possible.
                Query: {query}
                Answer in Hindi:
                """

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for Hindi literature. Respond appropriately based on the context provided."},
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )

                return response.choices[0].message.content
            except Exception as openai_e:
                print(f"OpenAI fallback also failed: {openai_e}")

        # If all LLM options fail, synthesize an answer from the documents
        if context_docs:
            # Create a synthesized answer from the most relevant documents
            synthesized_answer = f"प्रश्न: {query}\n\n"
            synthesized_answer += "संबंधित दस्तावेज़ से संग्रहीत जानकारी:\n\n"

            # Use the most relevant documents to synthesize an answer
            for i, doc in enumerate(context_docs[:3], 1):
                synthesized_answer += f"{i}. {doc['title']} - {doc['author']} (Score: {doc['score']:.3f})\n"
                # Include more text content for better synthesis
                text_preview = doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text']
                synthesized_answer += f"   सारांश: {text_preview}\n\n"

            # Add a synthesized conclusion based on the documents
            synthesized_answer += "उपरोक्त दस्तावेज़ के आधार पर, यह जानकारी इकट्ठा की गई है। कृपया विस्तृत जानकारी के लिए संबंधित दस्तावेज़ देखें।"
            return synthesized_answer
        else:
            return f"क्षमा करें, प्रश्न '{query}' के लिए कोई संबंधित दस्तावेज़ नहीं मिला।"

    def generate_poem(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate a Hindi poem based on the query and retrieved documents
        """
        # Find poetry-related documents to use as inspiration
        poetry_docs = [doc for doc in context_docs if doc.get('genre', '').lower() in ['poem', 'poetry', 'कविता', 'गीत']]

        if not poetry_docs:
            # If no poetry docs found, use any relevant docs
            poetry_docs = context_docs[:3] if context_docs else []

        # Create a prompt for poem generation
        if poetry_docs:
            inspiration = "\n".join([f"Title: {doc['title']}, Author: {doc['author']}, Excerpt: {doc['text'][:100]}"
                                   for doc in poetry_docs[:2]])
            prompt = f"""Based on the following poetry examples, create a new Hindi poem that captures similar essence, style, or theme:

Poetry Examples:
{inspiration}

Request: {query}

New Hindi Poem:"""
        else:
            prompt = f"""Create a beautiful Hindi poem based on the request: {query}

Hindi Poem:"""

        # Try to generate using the LLM
        try:
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            template = prompt
            prompt_template = PromptTemplate.from_template(template)

            # Create the chain
            chain = (
                {}  # Empty context since we're using the full prompt
                | prompt_template
                | self.llm
                | StrOutputParser()
            )

            # Generate response using the configured LLM
            response = chain.invoke({})

            # Check if response is meaningful - not just the prompt template
            if response and len(response.strip()) > 20 and not response.strip().startswith("Based on the following poetry examples") and not response.strip().startswith("Create a beautiful Hindi poem"):
                return f"आपके लिए एक नई कविता:\n\n{response}"
            else:
                # If LLM returns template or minimal response, create a synthesized poem from available content
                print(f"Poem generation returned template or minimal response: '{response[:50]}...'. Using synthesized approach.")
                return self.synthesize_poem_from_docs(query, poetry_docs)

        except Exception as e:
            print(f"Poem generation failed: {e}")
            # Fallback to synthesized poem
            return self.synthesize_poem_from_docs(query, poetry_docs)

    def synthesize_poem_from_docs(self, query: str, poetry_docs: List[Dict]) -> str:
        """
        Synthesize a poem from available poetry documents when LLM fails
        """
        if poetry_docs:
            # Create a poem by combining elements from different poems
            title = f"संग्रह से सार - {poetry_docs[0].get('title', 'कविता')}"
            author = poetry_docs[0].get('author', 'अज्ञात')

            # Extract lines from different poems to create a new one
            lines = []
            for doc in poetry_docs:
                text = doc.get('text', '')
                # Split into lines and take a few from each
                doc_lines = [line.strip() for line in text.split('\n') if line.strip()]
                # Take up to 2 lines from each document
                lines.extend(doc_lines[:2])

            # Limit to 6-8 lines for the new poem
            selected_lines = lines[:8] if lines else ["कविता लिखने में समस्या हुई", "कृपया पुनः प्रयास करें"]

            poem = "\n".join(selected_lines)

            return f"नई कविता: {title}\nलेखक: {author}\n\n{poem}\n\n(उपरोक्त कविता संग्रह में से तत्वों को जोड़कर बनाई गई है)"
        else:
            return f"कविता लिखने में समस्या हुई। कृपया कुछ समय बाद पुनः प्रयास करें।"

    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Main query method that retrieves documents and generates an answer
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(question, top_k)
        
        # Generate answer based on retrieved documents
        answer = self.generate_answer(question, relevant_docs)
        
        return {
            'answer': answer,
            'relevant_documents': relevant_docs
        }

# Example usage
if __name__ == "__main__":
    rag_system = HindiRAGSystem()
    
    # Example query
    question = "हिंदी कविता में प्रकृति का वर्णन कैसे किया जाता है?"
    result = rag_system.query(question)
    
    print("Question:", question)
    print("\nAnswer:", result['answer'])
    print("\nRelevant Documents:")
    for i, doc in enumerate(result['relevant_documents']):
        print(f"\n{i+1}. Title: {doc['title']}, Author: {doc['author']}")
        print(f"   Score: {doc['score']}")
        print(f"   Excerpt: {doc['text'][:200]}...")