import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
import pandas as pd
from langchain.prompts import PromptTemplate
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"


# This covers project rubric: Synthetic Data Generation->Generating Real Estate Listings with an LLM
def generate_synthetic_data(number_of_listings):
    try:

        model_name = "gpt-3.5-turbo"
        temperature = 0.7
        llm = OpenAI(model_name=model_name, temperature=temperature, max_tokens=4000)
        prompt_template = f"""Generate {number_of_listings} csv formatted real estate listings. Each listing must have these fields in the csv: Neighborhood (name),Price(in euros),Bedrooms (number of bedrooms),Bathrooms (number of bedrooms),House Size (in m2),Description (Creative house description),Neighborhood Description (creative neighborhood and surroundings description).
         Use real united states neighborhoods. Your response must be in english. csv format is a must.
         """
        return llm(prompt_template)
    except Exception as e:
        return f"An error occurred: {e}"

def save_synthdata():
    # Generate 10 real state listings
    synthetic_data = generate_synthetic_data(10)
    filename = "synthetic_listing.csv"
    print(synthetic_data)
    # Write to a file
    with open(filename, "w") as file:
        file.write(synthetic_data)


if __name__ == '__main__':
    # This covers project rubric: Synthetic Data Generation->Generating Real Estate Listings with an LLM
    #save_synthdata()
    loader = CSVLoader(file_path='./synthetic_listing.csv')
    loaded_listings = loader.load()
    # Read synthetic listings
    # Split documents in chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = splitter.split_documents(loaded_listings)
    # Define embedding class to be used in db
    embeddings = OpenAIEmbeddings()
    # Create db with synthetic data.
    # This cover Rubric: Semantic Search->Creating a Vector Database and Storing Listings
    db = Chroma.from_documents(split_docs, embeddings)
    # Collect user preferences
    questions = [
        "How big do you want your house to be?",
        "What are 2 most important things for you regarding the location of the property?",
        "Which amenities would you like?",
        "Which transportation options are important to you?",
        "How urban do you want your neighborhood to be?",
    ]
    answers = [
        "It must have more than 2 bedrooms",
        "It must be nearby the beach, a lake, or a river. Must have sunny weather",
        "A well equipped kitchen and terrace will be nice",
        "Bike friendly, well connected with the highway",
        "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."]
    q_and_a = "\n".join([f"ai assistant: {question} \n human answer: {answer} \n" for question, answer in zip(questions, answers)])
    print(q_and_a)

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0, max_tokens=4000)
    n_interesting_listings = 5
    n_final_listings = 3
    query = f"""Based on the next conversation between an ai assistant and a human, and the real estate listings in the context.
        Tell me which {n_final_listings} listings fit the best to the human interests extracted from the questions ans answers. Make sure you
        do not paraphrase the listings, only use the information provided in the listings and take into account human preferences.
        Return the listings in descending order, where the first one is the best fit to human preferences.
        The conversation is the next one:
        {q_and_a}
        """
    # This cover project rubric: Semantic Search->Semantic Search of Listings Based on Buyer Preferences
    # Query database for listings similar to buyer's preferences.
    context_listings = db.similarity_search(query, k=n_interesting_listings)
    #print([doc.page_content.split("\n")[0] for doc in similar_listings])
    prompt = PromptTemplate(template="{query}\nContext: {context}",input_variables=["query", "context"])
    chain = load_qa_chain(llm, prompt=prompt, chain_type="stuff")
    # This cover project rubric: Semantic Search->Semantic Search of Listings Based on Buyer Preferences
    # Refine search with llm
    interesting_listings = chain.run(input_documents=context_listings, query=query)
    with open("out_listings_before_augmentation.txt","w") as file_io:
        file_io.write(interesting_listings)

    final_listings_query =f""" You will receive {n_final_listings} real estate listings and a conversation between an ai assistant
     and a human that provides information about the human's real state preferences. For each input listing, 
     augment it's Description and Neighborhood description, tailoring it to resonate with the human’s specific preferences.
     Subtly  emphasize aspects of the property that align with what the human is looking for.
     Ensure that the augmentation process enhances the appeal of the listing without altering factual information.
     The descriptions should be unique, appealing, and tailored to the preferences provided.
     Here are the input listings:\n
     {interesting_listings}
     Here is the conversation between the ai assistant and the human:
     {q_and_a}
     
     """
    # This Covers project rubric: Augmented Response Generation -> Logic for Searching and Augmenting Listing Descriptions,
    # Use of LLM for Generating Personalized Descriptions augmentation
    final_listings = llm(final_listings_query,temperature=0.7)
    with open("out_augmented_listings.txt","w") as file_io:
        file_io.write(final_listings)

    print(final_listings)
