# **MediLexica: Your AI powered Medical Dictionary**

![logo](https://github.com/user-attachments/assets/9c75ff8d-0181-44c5-bfe5-3eff73eea38d)


MediLexica is an AI powered **medical dictionary** designed to help users understand complex medical terms and concepts. It is a side project of **EarlyMed**, a platform dedicated to empowering users with medical knowledge before they consult a doctor. MediLexica leverages advanced natural language processing (NLP) and retrieval-augmented generation (RAG) techniques to provide accurate, detailed, and empathetic responses to medical queries.

**Try MediLexica on Hugging Face Spaces:** [MediLexica Deployment](https://huggingface.co/spaces/MahatirTusher/MediLexica)

![Medilexica UI](https://github.com/user-attachments/assets/23bb5148-d2fd-4a44-92e0-eeed8bc9f3b7)


---

## **Table of Contents**
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [How It Works](#how-it-works)
4. [Technologies Used](#technologies-used)
5. [Workflow](#workflow)
6. [Setup and Deployment](#setup-and-deployment)
7. [Future Enhancements](#future-enhancements)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgements](#acknowledgements)

---

## **Overview**

MediLexica is part of the **EarlyMed Project**, which aims to make medical knowledge accessible to everyone. Often, patients feel overwhelmed by the complex medical jargon used by healthcare professionals. MediLexica bridges this gap by providing a user-friendly interface where users can ask medical-related questions and receive accurate, detailed answers with citations from trusted medical resources.

The app uses a **PDF (`Medilexica.pdf`)** as its primary knowledge source, which includes content from three authoritative medical dictionaries:
1. **Webster's New World Medical Dictionary**
2. **Dictionary of Medical Terms**
3. **AMA Glossary of Medical Terms**

MediLexica is deployed on **Hugging Face Spaces**, making it accessible to users worldwide.

---

## **Key Features**

### **1. Accurate Medical Information**
- MediLexica provides **accurate and detailed responses** to medical queries, grounded in trusted medical literature.

### **2. Source Citations**
- Every response includes **citations from the `Medilexica.pdf` file**, ensuring transparency and credibility.
- Example:
  ```
  Medilexica: The treatment for severe aortic stenosis involves valve replacement surgery.

  Sources:
  1. Page 45 of 'Medilexica.pdf': Aortic stenosis is a condition where the aortic valve...
  ```

### **3. User-Friendly Interface**
- The **Gradio-based interface** is simple and intuitive, with clear instructions and a clean design.

### **4. Error Handling**
- The app gracefully handles errors and provides meaningful feedback to the user.
- Example:
  ```
  Error: PDF file not found at: Medilexica.pdf
  ```

### **5. Fast and Efficient**
- The app uses **ChromaDB** for efficient retrieval of relevant information and **Groq API** for high-performance inference.

---

## **How It Works**

### **1. PDF Processing**
- The `Medilexica.pdf` file is loaded using `PyPDFLoader`.
- The content is split into smaller chunks using `RecursiveCharacterTextSplitter` for efficient processing.

### **2. Vector Database (ChromaDB)**
- The text chunks are embedded into a vector space using `HuggingFaceEmbeddings`.
- The embeddings are stored in a **ChromaDB** instance for fast and accurate retrieval.

### **3. Retrieval-Augmented Generation (RAG)**
- When a user query is received, the app retrieves the most relevant chunks from the ChromaDB.
- The **Groq language model** (`llama-3.3-70b-versatile`) generates a response based on the retrieved chunks and the user query.

### **4. Gradio Interface**
- The user interacts with the app through a **Gradio interface**, which includes:
  - A logo (`logo.png`) displayed at the top center.
  - A text input box for entering medical-related queries.
  - A text output box for displaying responses.
  - A submit button to trigger the query.

---

## **Technologies Used**

### **1. LangChain**
- Used for document loading, text splitting, embeddings, retrieval, and question-answering.
- Key Modules:
  - `PyPDFLoader`: Loads the PDF file.
  - `RecursiveCharacterTextSplitter`: Splits the PDF content into smaller chunks.
  - `HuggingFaceEmbeddings`: Generates embeddings for the text chunks.
  - `Chroma`: Manages the vector database.
  - `RetrievalQA`: Combines retrieval and generation for question-answering.

### **2. Groq API**
- Powers the language model (`llama-3.3-70b-versatile`) used for generating responses.
- Features:
  - High-performance inference.
  - Deterministic responses (temperature = 0).

### **3. Gradio**
- Used to build the user interface for the app.
- Features:
  - Easy-to-use interface components (text input, text output, buttons).
  - Real-time interaction with the backend.

### **4. Hugging Face Spaces**
- The app is deployed on **Hugging Face Spaces**, making it accessible to users worldwide.

---

## **Workflow**

### **Step 1: Initialization**
- The app initializes the Groq language model and sets up the ChromaDB.
- If the ChromaDB does not exist, it is created by processing the `Medilexica.pdf` file.

### **Step 2: User Interaction**
- The user enters a medical-related query in the Gradio interface.
- The query is passed to the `medilexica_query()` function.

### **Step 3: Query Processing**
- The `RetrievalQA` chain retrieves relevant chunks from the ChromaDB.
- The Groq language model generates a response based on the retrieved chunks and the user query.

### **Step 4: Response Formatting**
- The response is formatted to include:
  - The model's answer.
  - Source citations (page numbers and snippets from `Medilexica.pdf`).

### **Step 5: Display Results**
- The formatted response is displayed in the Gradio interface.

---

## **Future Enhancements**

### **1. Expand Knowledge Base**
- Include more medical resources to cover a wider range of topics.

### **2. Multilingual Support**
- Add support for multiple languages to make the app accessible to a global audience.

### **3. Interactive Features**
- Introduce features like **voice input** and **interactive Q&A** for a more engaging user experience.

### **4. Mobile App**
- Develop a mobile application for on-the-go access to MediLexica.

### **5. Bengali Version**
- A bengali version of MediLexica is going to be released soon. It will generate responses in fine, well structured bengali, with a bengali UI. You will get its source code in the MediLexica.ipynb notebook.

![bengali UI](https://github.com/user-attachments/assets/d5833746-0101-43f8-80df-38bacb1c7e04)


---

## **Contributing**

We welcome contributions to MediLexica! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

---

## **License**

MediLexica is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## **Acknowledgements**
- **Hugging Face**: For providing the platform to deploy MediLexica.
- **Groq**: For their high-performance language model API.
- **LangChain**: For their powerful tools for building NLP applications.
