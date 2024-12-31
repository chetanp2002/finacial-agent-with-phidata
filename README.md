![My Image](https://raw.githubusercontent.com/chetanp2002/images/main/finance_agent.png)


# Financial Chatbot using Streamlit

This project implements a **Financial Chatbot** using the **PHI** library, **Groq** model, and **YFinanceTools** for querying stock data and providing responses. The chatbot allows users to ask about stock prices and retrieve information in a clear and structured format.

## Technologies Used

- **Streamlit**: For building the interactive web application.
- **PHI**: A framework for interacting with AI models, used here with the **Groq** model.
- **Groq Model**: A powerful model used for processing user queries.
- **YFinanceTools**: A tool to fetch stock price data via Yahoo Finance API.
- **dotenv**: For securely loading API keys from environment files.

## Features

- Allows users to ask about stock prices through a chatbot interface.
- Displays real-time stock price data in a clear format.
- Simple and interactive interface built with Streamlit.

## Requirements

- Python 3.7+
- Streamlit
- PHI library
- Groq model
- YFinanceTools
- dotenv
- Yahoo Finance API key (loaded through `.env` file)

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/chetanp2002/Python.git
    cd Python
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up your `.env` file:**

    Create a `.env` file in the root directory of the project and add your PHI API key:
    ```plaintext
    PHI_API_KEY=your_api_key_here
    ```

4. **Run the application:**
    ```bash
    streamlit run app.py
    ```

    The app will start and you can interact with the financial chatbot.

## Usage

- Once the application is running, open the provided URL in your browser.
- Type your query about stock prices in the input field.
- The chatbot will provide stock data from Yahoo Finance, presented in a readable format.

## Example

1. Ask about stock prices like:
    - "What is the price of Apple stock?"
    - "Tell me the current price of Tesla."
   
2. The chatbot will respond with the latest stock data for the requested company.

## Error Handling

- If the agent fails to initialize, an error message will be displayed.
- If there is an issue fetching stock data, appropriate error messages will be shown.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [PHI](https://www.phi.ai) for providing the agent and model.
- [Streamlit](https://streamlit.io) for creating a simple and powerful interface.
- [YFinanceTools](https://pypi.org/project/yfinance-tools/) for fetching stock data.
