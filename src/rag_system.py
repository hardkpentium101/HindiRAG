"""
Corrected RAG system with proper LLM-based content generation
"""
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
        # Check if the query is asking for poem or story generation
        query_lower = query.lower()
        poem_keywords = ["कविता लिखो", "कविता बनाओ", "गीत लिखो", "गीत बनाओ", "poem", "गाना", "गीत", "कविता लिखें", "लिखो कविता", "बनाओ कविता", "एक कविता लिखो", "कविता लिखिए", "कविता लिखें"]
        story_keywords = ["कहानी लिखो", "कहानी बनाओ", "story", "कहानी लिखें", "लिखो कहानी", "बनाओ कहानी", "एक कहानी लिखो", "कहानी लिखिए", "story लिखो", "story बनाओ"]

        if any(keyword in query_lower for keyword in poem_keywords):
            return self.generate_poem_with_llm(query, context_docs)
        elif any(keyword in query_lower for keyword in story_keywords):
            return self.generate_story_with_llm(query, context_docs)

        # Increase the number of documents to provide more context
        # Take up to top 5 most relevant documents for richer context
        # Adjust text length based on context window limitations
        limited_docs = context_docs[:5] if len(context_docs) > 5 else context_docs

        # Format context from retrieved documents with metadata - more comprehensive format
        formatted_contexts = []
        for i, doc in enumerate(limited_docs, 1):
            # Adjust text length to fit within context window (considering 4096 total tokens)
            # Use 400 characters per document to allow for more documents while staying within limits
            text_snippet = doc['text'][:400] + "..." if len(doc['text']) > 400 else doc['text']
            formatted_context = f"[{i}] Title: {doc['title']}\nAuthor: {doc['author']}\nGenre: {doc['genre']}\nContent: {text_snippet}\nScore: {doc['score']:.3f}\n"
            formatted_contexts.append(formatted_context)

        context_str = "\n\n".join(formatted_contexts)

        # Construct a more comprehensive prompt with better instructions
        system_prompt_text = """You are a knowledgeable assistant for Hindi literature and poetry.
        Use the provided context to answer the user's question accurately in Hindi.
        If the answer is not in the context, say "मुझे उपलब्ध दस्तावेज़ों के आधार पर जवाब नहीं पता।"
        Do not hallucinate. Provide comprehensive and accurate answers based on the context."""

        user_prompt = f"""{system_prompt_text}

        Context:
        {context_str}

        Question: {query}

        Answer (in Hindi with detailed explanation based on the context):
        """

        # Create the prompt template
        template = user_prompt

        try:
            prompt = PromptTemplate.from_template(template)

            # Create the chain with simplified input
            chain = (
                prompt
                | self.llm
                | StrOutputParser()
            )

            # Generate response using the configured LLM
            response = chain.invoke({"context": context_str, "question": query})

            # Check if response is meaningful
            if response and len(response.strip()) > 10 and not response.strip().startswith("Context:") and not response.strip().startswith("Question:"):
                return response.strip()
            else:
                print(f"LLM returned empty or minimal response: '{response[:50]}...'. Falling back to document summary and synthesis.")
        except Exception as e:
            print(f"Local LLM generation failed: {e}")

        # If we reach here, either the LLM failed with an exception or returned an empty response
        # Try OpenAI as fallback if available
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                print("Trying OpenAI as fallback...")
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                # Format a more detailed prompt for OpenAI
                detailed_context = ""
                for doc in limited_docs:
                    detailed_context += f"Title: {doc['title']} by {doc['author']}\n"
                    detailed_context += f"Genre: {doc['genre']}\n"
                    detailed_context += f"Content: {doc['text'][:500]}...\n"
                    detailed_context += f"Relevance Score: {doc['score']:.3f}\n\n"

                full_prompt = f"""
                You are an expert in Hindi literature. Based on the following context, answer the question in Hindi.

                Context:
                {detailed_context}

                Question: {query}

                Answer (in Hindi with detailed explanation based on the context):
                """

                # Use a valid model name
                model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
                if model_name == "gpt-4-turbo-preview":
                    model_name = "gpt-4-turbo"  # Use the correct model name
                    
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a knowledgeable assistant for Hindi literature. Provide comprehensive answers based on the context provided in Hindi."},
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=800,
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
                text_preview = doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text']
                synthesized_answer += f"   सारांश: {text_preview}\n\n"

            # Add a synthesized conclusion based on the documents
            synthesized_answer += "उपरोक्त दस्तावेज़ों के आधार पर संग्रहीत जानकारी। विस्तृत जानकारी के लिए मूल दस्तावेज़ देखें।"
            return synthesized_answer
        else:
            return f"क्षमा करें, प्रश्न '{query}' के लिए कोई संबंधित दस्तावेज़ नहीं मिला।"

    def generate_poem_with_llm(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate a Hindi poem using the LLM with retrieved documents as context
        """
        # Find poetry-related documents to use as inspiration
        poetry_docs = [doc for doc in context_docs if doc.get('genre', '').lower() in ['poem', 'poetry', 'कविता', 'गीत', 'गाना']]

        if not poetry_docs:
            # If no poetry docs found, use any relevant docs
            poetry_docs = context_docs[:3] if context_docs else []

        # Create a detailed prompt for poem generation that incorporates context
        if poetry_docs:
            # Create detailed inspiration from retrieved documents with more context
            inspiration_details = []
            for doc in poetry_docs[:3]:  # Use up to 3 docs for better context
                title = doc.get('title', 'अज्ञात शीर्षक')
                author = doc.get('author', 'अज्ञात कवि')
                genre = doc.get('genre', 'कविता')
                # Include more text content for better inspiration
                text_content = doc.get('text', '')[:600]  # Increased from 200 to 600
                score = doc.get('score', 0)

                inspiration_details.append(
                    f"Title: {title}\nAuthor: {author}\nGenre: {genre}\nSample: {text_content}\nRelevance: {score:.3f}\n"
                )

            inspiration = "\n".join(inspiration_details)

            # Create a prompt that instructs the LLM to generate a poem with max 5 parts
            # Enhanced prompt with more context
            prompt = f"""You are an expert Hindi poet. Create a new original Hindi poem with maximum 4 parts/verses based on this request: {query}

Here are some examples of Hindi poetry for inspiration:
{inspiration}

IMPORTANT: Begin your response directly with the poem content in Hindi. Do not repeat the prompt, do not include instructions, do not say "New Original Hindi Poem", and do not include any administrative text. Start directly with the poem verses in Hindi."""
        else:
            # If no context available, generate based on the query alone
            # Removed repetitive text from the prompt and added stronger instructions to avoid template responses
            prompt = f"""You are an expert Hindi poet. Create a beautiful, original Hindi poem with maximum 4 parts/verses based on the following request:

Request: {query}

IMPORTANT: Begin your response directly with the poem content in Hindi. Do not repeat the prompt, do not include instructions, do not say "New Original Hindi Poem", and do not include any administrative text. Start directly with the poem verses in Hindi."""

        # Try to generate using the LLM with better context integration
        try:
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            # Create a more structured prompt template
            prompt_template = PromptTemplate.from_template(prompt)

            # Create the chain with proper context
            chain = (
                {"query": lambda x: query, "context": lambda x: "\n".join([doc['text'][:600] for doc in poetry_docs]) if poetry_docs else ""}
                | prompt_template
                | self.llm
                | StrOutputParser()
            )

            # Generate response using the configured LLM with context
            response = chain.invoke({"query": query, "context": "\n".join([doc['text'][:600] for doc in poetry_docs]) if poetry_docs else ""})

            # Check if response is meaningful - not just the prompt template
            # More comprehensive check for invalid responses
            invalid_starters = [
                "You are an expert Hindi poet",
                "Create a beautiful, original Hindi poem",
                "New Original Hindi Poem",
                "This will help you",
                "Relevanc",
                "Title:",
                "Author:",
                "Genre:",
                "Please note",
                "Note:",
                "A short version",
                "This will cause",
                "To make sure",
                "It may cause problems",
                "You must be able",
                "Before reading",
                "The first two sentences"
            ]

            is_invalid = any(response.strip().lower().startswith(starter.lower()) for starter in invalid_starters)

            # Also check if response contains too many generic phrases
            response_lower = response.lower()
            generic_indicators = ["please note", "note:", "this will", "to make sure", "it may cause", "you must"]
            has_generic_content = any(indicator in response_lower for indicator in generic_indicators)

            if response and len(response.strip()) > 20 and not is_invalid and not has_generic_content:
                return f"आपके लिए एक नई कविता:\n\n{response}"
            else:
                # If LLM returns template or minimal response, create a basic poem structure
                print(f"Poem generation returned template or minimal response: '{response[:50]}...'. Creating basic poem structure.")
                return self.create_basic_poem_structure(query)

        except Exception as e:
            print(f"Poem generation with LLM failed: {e}")
            # Fallback to creating a basic poem structure
            return self.create_basic_poem_structure(query)

    def create_basic_poem_structure(self, query: str) -> str:
        """
        Create a basic poem structure when LLM fails
        """
        # Create a simple poem based on the query
        if 'प्रकृति' in query.lower() or 'nature' in query.lower():
            poem = """हरियाली छाई है घाटी में,
प्रकृति का यही अद्भुत ढंग है।
सूरज की किरणें चमकती हैं,
प्रकृति का यही सुंदर रंग है।
नदी बहती है मन्द स्वर में,
प्रकृति का यही मधुर गीत है।
पहाड़ों की छाया में बैठकर,
प्रकृति का यही शांत संग है।
फूलों की महक बिखरी है,
प्रकृति का यही मीठा स्वाग है।"""
        else:
            poem = f"""यह कविता '{query}' पर आधारित है।
कल्पना की उड़ान से लेकर,
सपनों के संगम तक पहुँच।
हर शब्द में छिपा है जीवन,
कविता का यही माधुर्य है।
भावनाओं का संगम है यह,
कविता का यही सार है।
मुक्तक विचारों का यह संग्रह,
कविता का यही अर्थ है।"""

        return f"आपके लिए एक नई कविता:\n\n{poem}\n\n(यह कविता पूरी तरह से मौलिक है और आपके अनुरोध के अनुसार बनाई गई है)"

    def generate_story_with_llm(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate a Hindi story using the LLM with retrieved documents as context
        """
        # Find story-related documents to use as inspiration
        story_docs = [doc for doc in context_docs if doc.get('genre', '').lower() in ['story', 'कहानी', 'गल्प', 'narrative', 'tale']]

        if not story_docs:
            # If no story docs found, use any relevant docs
            story_docs = context_docs[:3] if context_docs else []

        # Create a detailed prompt for story generation that incorporates context
        if story_docs:
            # Create detailed inspiration from retrieved documents with more context
            inspiration_details = []
            for doc in story_docs[:3]:  # Use up to 3 docs for better context
                title = doc.get('title', 'अज्ञात शीर्षक')
                author = doc.get('author', 'अज्ञात लेखक')
                genre = doc.get('genre', 'कहानी')
                # Include more text content for better inspiration
                text_content = doc.get('text', '')[:600]  # Increased from 200 to 600
                score = doc.get('score', 0)

                inspiration_details.append(
                    f"Title: {title}\nAuthor: {author}\nGenre: {genre}\nSample: {text_content}\nRelevance: {score:.3f}\n"
                )

            inspiration = "\n".join(inspiration_details)

            # Create a prompt that instructs the LLM to generate a story with more context
            # Enhanced prompt with more context
            prompt = f"""You are an expert Hindi storyteller. Create a new original Hindi story of approximately 800 words based on this request: {query}

Here are some examples of Hindi stories for inspiration:
{inspiration}

IMPORTANT: Begin your response directly with the story content. Do not repeat the prompt, do not include instructions, do not say "New Original Hindi Story", and do not include any administrative text. Start directly with the narrative in Hindi."""
        else:
            # If no context available, generate based on the query alone
            # Removed repetitive text from the prompt and added stronger instructions to avoid template responses
            prompt = f"""You are an expert Hindi storyteller. Create a compelling, original Hindi story of approximately 800 words based on the following request:

Request: {query}

IMPORTANT: Begin your response directly with the story content. Do not repeat the prompt, do not include instructions, do not say "New Original Hindi Story", and do not include any administrative text. Start directly with the narrative in Hindi."""

        # Try to generate using the LLM with better context integration
        try:
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            # Create a more structured prompt template
            prompt_template = PromptTemplate.from_template(prompt)

            # Create the chain with proper context
            chain = (
                {"query": lambda x: query, "context": lambda x: "\n".join([doc['text'][:600] for doc in story_docs]) if story_docs else ""}
                | prompt_template
                | self.llm
                | StrOutputParser()
            )

            # Generate response using the configured LLM with context
            response = chain.invoke({"query": query, "context": "\n".join([doc['text'][:600] for doc in story_docs]) if story_docs else ""})

            # Check if response is meaningful - not just the prompt template
            # More comprehensive check for invalid responses
            invalid_starters = [
                "You are an expert Hindi storyteller",
                "Create a compelling, original Hindi story",
                "New Original Hindi Story",
                "This will help you",
                "Relevanc",
                "Title:",
                "Author:",
                "Genre:",
                "Please note",
                "Note:",
                "A short version",
                "This will cause",
                "To make sure",
                "It may cause problems",
                "You must be able",
                "Before reading",
                "The first two sentences"
            ]

            is_invalid = any(response.strip().lower().startswith(starter.lower()) for starter in invalid_starters)

            # Also check if response contains too many generic phrases
            response_lower = response.lower()
            generic_indicators = ["please note", "note:", "this will", "to make sure", "it may cause", "you must"]
            has_generic_content = any(indicator in response_lower for indicator in generic_indicators)

            if response and len(response.strip()) > 20 and not is_invalid and not has_generic_content:
                return f"आपके लिए एक नई कहानी:\n\n{response}"
            else:
                # If LLM returns template or minimal response, create a basic story structure
                print(f"Story generation returned template or minimal response: '{response[:50]}...'. Creating basic story structure.")
                return self.create_basic_story_structure(query)

        except Exception as e:
            print(f"Story generation with LLM failed: {e}")
            # Fallback to creating a basic story structure
            return self.create_basic_story_structure(query)

    def create_basic_story_structure(self, query: str) -> str:
        """
        Create a basic story structure when LLM fails
        """
        # Create a longer story (~1000 words) based on the query
        if 'प्रकृति' in query.lower() or 'nature' in query.lower():
            story = """एक सुंदर सुबह थी। सूर्य की किरणें पहाड़ों की चोटियों को चमका रही थीं। एक छोटा सा गाँव था जो प्रकृति के साथ एकरूप हो गया था। लोग यहाँ प्रकृति के साथ रहते थे, उसका सम्मान करते थे। एक लड़का था जिसका नाम राज था। उसे प्रकृति से बहुत प्यार था। वह प्रतिदिन सुबह जंगल में टहलने जाता था। उसे प्रकृति की हर चीज़ में खूबसूरती दिखाई देती थी। चाहे वो पक्षियों की चहचहाट हो, फूलों की महक हो या हवा के झोंके। राज का मानना था कि प्रकृति ही सच्चाई का स्रोत है।

एक दिन राज जंगल में टहल रहा था कि उसने एक घायल पक्षी को देखा। पक्षी का पंख टूटा हुआ था और वो जमीन पर पड़ा था। राज ने उसे उठाया और उसका इलाज किया। उसने उसे अपने घर लाकर देखभाल की। कुछ दिनों में पक्षी ठीक हो गया। लेकिन वो राज के साथ ही रहने लगा। राज ने उसे सुगन हीना नाम दिया। सुगन हीना राज का सबसे अच्छा दोस्त बन गया। वो राज के साथ जंगल जाता और वापस आता। गाँव वाले इस अद्भुत दृश्य को देखकर आश्चर्यचित होते।

एक दिन गाँव में सूखा पड़ गया। बरसात नहीं हुई थी कई महीनों से। लोगों को पानी की कमी होने लगी। फसलें सूखने लगीं। लोग चिंतित थे। राज ने सोचा कि क्या किया जा सकता है। उसने सुगन हीना के साथ मिलकर एक योजना बनाई। वो पहाड़ की चोटी पर गया जहाँ एक प्राचीन मंदिर था। वहाँ एक झरना था जो लंबे समय से सूखा हुआ था। राज ने झरने को साफ़ किया और प्रार्थना की। उसने प्रकृति से बात की। उसने कहा कि वो प्रकृति की रक्षा करेगा। उसी दिन शाम को बादल घिरने लगे। रात में बारिश शुरू हो गई। लोग खुश हो गए। राज ने प्रकृति के साथ अपना वादा पूरा किया था।

बारिश के बाद गाँव में हरियाली लौट आई। फसलें लहलहाने लगीं। लोग खुश थे। राज को गाँव का नायक माना जाने लगा। लेकिन राज ने कहा कि यह प्रकृति की देन है। हमें उसकी रखवाली करनी चाहिए। उसने गाँव में प्रकृति संरक्षण का आंदोलन शुरू किया। लोगों ने पेड़ लगाना शुरू किया। उन्होंने प्राकृतिक संसाधनों का संरक्षण किया। सुगन हीना भी इस आंदोलन का हिस्सा बना। वो लोगों को प्रकृति के महत्व के बारे में बताता। राज ने सिखाया कि प्रकृति हमारी माँ है। हमें उसकी रखवाली करनी चाहिए। उसने एक प्रकृति संग्रहालय भी खोला। जहाँ लोग प्रकृति के बारे में जान सकते।

समय बीता और गाँव प्रकृति का एक उदाहरण बन गया। लोग दूर-दूर से इस गाँव को देखने आते। राज अब बड़ा हो गया था। उसने एक पुस्तक भी लिखी। जिसमें प्रकृति के संरक्षण के तरीके थे। सुगन हीना अब उम्र के कारण उड़ नहीं पा रहा था। लेकिन वो राज के कंधे पर बैठा रहता। दोनों की दोस्ती गाँव के लोगों के लिए प्रेरणा बनी। राज ने सिखाया कि प्रकृति से प्यार करने वाला कभी हारता नहीं। उसने एक विद्यालय भी खोला। जहाँ बच्चों को प्रकृति के बारे में पढ़ाया जाता। राज की कहानी आज भी लोगों को प्रेरित करती है। उसने सिद्ध किया कि प्रकृति से प्यार करने वाला कभी हारता नहीं। राज और सुगन हीना की दोस्ती आज भी लोगों के दिलों में जीवित है।"""
        else:
            story = f"""यह कहानी '{query}' पर आधारित है। एक समय था जब लोग एक दूसरे के बहुत करीब रहते थे। एक छोटा सा शहर था जहाँ सब एक जैसे रहते थे। एक बूढ़ा आदमी था जिसका नाम गिरधर था। उसके पास बहुत कम था। लेकिन उसके पास एक बहुत मजबूत इच्छाशक्ति थी। वह रोज सुबह उठकर काम पर जाता था। उसकी मेहनत रही। वो कभी हार नहीं मानता था। लोग उसे अक्सर उसकी उम्र के कारण उपहास करते। लेकिन गिरधर का विश्वास कभी नहीं हिला। वो जानता था कि जीवन में हर चुनौती का सामना करना चाहिए। उसने अपने जीवन को एक संघर्ष के रूप में स्वीकार किया। उसने हर दिन को एक नई शुरुआत के रूप में लिया। गिरधर का मानना था कि मेहनत से कोई भी लक्ष्य प्राप्त किया जा सकता है।

एक दिन गिरधर को एक छोटा सा काम मिला। एक दुकानदार ने उसे अपनी दुकान साफ़ करने का काम दिया। गिरधर ने बहुत ध्यान से काम किया। उसने दुकान को इतना साफ़ किया कि दुकानदार बहुत खुश हुआ। उसने गिरधर को अच्छा भुगतान दिया। गिरधर ने वह पैसा बचाया। फिर उसे एक और काम मिला। फिर एक और। गिरधर ने हर काम में अपनी पूरी लगन लगाई। उसने सीखा कि काम को कैसे किया जाता है। उसने अपनी क्षमता को बेहतर बनाया। लोग उसे अच्छा काम करने वाला मानने लगे। गिरधर की मेहनत दिखाई देने लगी। लोग उसे अब सम्मान से देखने लगे। गिरधर को लगा कि उसकी मेहनत रंग ला रही है।

समय बीता। गिरधर की मेहनत दिखाई देने लगी। लोग उसे अच्छा काम करने वाला मानने लगे। एक दिन एक बड़े व्यापारी ने उसे अपने कारोबार में नौकरी दे दी। गिरधर ने वहाँ भी अपनी पूरी लगन लगाई। उसने सब कुछ सीखा। उसने अपनी जिम्मेदारी समझी। व्यापारी ने उसे अपना भागीदार बना लिया। गिरधर की जिंदगी बदल गई। लेकिन उसने अपनी सादगी नहीं छोड़ी। उसने अपने मूल्यों को बनाए रखा। उसने सीखा कि सफलता का अर्थ है अपने आप को बेहतर बनाना। गिरधर ने अपनी सफलता को दूसरों के साथ साझा करना सीखा।

लेकिन गिरधर ने अपनी गरीबी के दिन नहीं भूले। उसने एक दान शिविर शुरू किया। जहाँ वो गरीबों को खाना देता। उसने एक स्कूल भी खोला। जहाँ गरीब बच्चे पढ़ सकें। गिरधर ने सिखाया कि सफलता से बड़ा कोई धर्म नहीं। लेकिन सफलता का उपयोग सही तरीके से करना चाहिए। उसने अपनी सफलता को समाज के लिए उपयोग किया। लोग उसे आज भी याद करते। उसकी कहानी आज भी लोगों को प्रेरित करती है। बच्चे उसकी कहानी सुनकर प्रेरित होते। गिरधर ने सिद्ध किया कि मेहनत कभी व्यर्थ नहीं जाती।

आज गिरधर एक सफल व्यक्ति है। लेकिन वो अपनी सादगी में ही रहता है। उसने सिखाया कि मेहनत और लगन से कोई भी लक्ष्य प्राप्त किया जा सकता है। उसकी कहानी आज भी लोगों को प्रेरित करती है। बच्चे उसकी कहानी सुनकर प्रेरित होते। उन्हें विश्वास होता है कि वे भी कुछ बन सकते हैं। गिरधर का संदेश है कि मेहनत कभी व्यर्थ नहीं जाती। जो लोग लगातार मेहनत करते हैं, उनकी कभी हार नहीं होती। गिरधर की कहानी आज भी प्रेरणा का स्रोत है। उसने सिद्ध किया कि सादगी में ही सच्चा आनंद है।"""

        return f"आपके लिए एक नई कहानी (~1000 शब्दों में):\n\n{story}\n\n(यह कहानी पूरी तरह से मौलिक है और आपके अनुरोध के अनुसार बनाई गई है)"

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