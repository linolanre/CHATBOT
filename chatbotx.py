import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import streamlit as st

# st.title("Chatbot Demo")
with open('Samsung Dialog.txt', 'r', encoding='utf8', errors='ignore') as file: 
    dataset = file.read()

sent_tokens = nltk.sent_tokenize(dataset)
word_tokens = nltk.word_tokenize(dataset)
lemmatizer = nltk.stem.WordNetLemmatizer()

def preprocess(tokens):
    return [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]

corpus = [" ".join(preprocess(nltk.word_tokenize(sentence))) for sentence in sent_tokens]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

def chatbot_response(user_input):
    user_input = " ".join(preprocess(nltk.word_tokenize(user_input)))

    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, X)
    idx = similarities.argmax()
    return sent_tokens[idx]


# ---------------------------------- STREAMLIT IMPLEMENTATION --------------------------------------
import streamlit as st


st.title("CHATBOT MACHINE.")
st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")

quit_sentences = ['quit', 'bye', 'Goodbye', 'exit']

history = []

st.markdown('<h3>Quit Words are: Quit, Bye, Goodbye, Exit</h3>', unsafe_allow_html = True)

# Get the user's question    
user_input = st.text_input(f'Input your response')
if user_input not in quit_sentences:
    if st.button("Submit"):
        # Call the chatbot function with the question and display the response
        response = chatbot_response(user_input)
        st.write("Chatbot: " + response)

        # Create a history for the chat
        history.append(('User: ', user_input))
        history.append(('Bot: ', chatbot_response(user_input)))
else:
    st.write('Bye')

st.markdown('<hr><hr>', unsafe_allow_html= True)
st.subheader('Chat History')

chat_history_str = '\n'.join([f'{sender}: {message}' for sender, message in history])

st.text_area('Conversation', value=chat_history_str, height=300)

# st.write("Welcome to the chatbot! How can I help you today?")


    # while True:
    #     user_input = st.text_input(f'Input your response')
    #     if user_input.lower() == 'quit':
    #         break
    #     response = chatbot_response(user_input)
    #     st.write("Bot Response:", response)



# def preprocess(text):
#     # tokenize the text into sentences
# sentences = sent_tokenize(text)
    
#     # remove stopwords and punctuation from each sentence
# filtered_sentences = []
# stop_words = set(stopwords.words('english'))

# for sentence in sentences:
# # tokenize the sentence into words
#  stop_words = word_tokenize(sentence)
# # remove stop words and punctuation
# filtered_words = [word.lower() 
# for word in words if word.lower() not in stop_words and word.isalnum()]
#         # join the filtered words into a string
# filtered_sentence = "".join(filtered_words)
# filtered_sentences.append(filtered_sentence)
        
# return filtered_sentences