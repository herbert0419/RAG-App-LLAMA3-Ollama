# RAG-App-LLAMA3-Ollama

## Running the Streamlit application

1. **Clone repo**: Run this in your terminal 

      ```bash
      git clone https://github.com/herbert0419/RAG-App-LLAMA3-Ollama.git
      ```

2. **Install Dependencies**: Execute to install dependencies
  
      ```bash
      pip install -r requirements.txt
      ```

3. **Launch the App**: Run to start the Streamlit interface on `localhost`

      ```bash
      streamlit run app.py
      ``` 

4. **Running Terminal in Colab**: Execute
      ```bash
      !pip install colab-xterm
      ```

5. **Load the installed Terminal**: Execute
      ```bash
      %load_ext colabxterm
      ```

6. **Run the loaded Terminal**: Execute
      ```bash
      %xterm
      ```

7. **Install ollama in colab terminal**: Execute
      ```bash
      curl -fsSL https://ollama.com/install.sh | sh
      ```

8. **Run & serve Ollama**: Execute
      ```bash
      ollama serve
      ```

      ```bash
      ollama pull llama3:latest
      ```