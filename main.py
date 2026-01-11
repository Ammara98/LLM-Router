from dotenv import load_dotenv
from providers import GeminiProvider, GroqProvider
from router import LLMRouter

load_dotenv()


def main():

    try:
        provider = GeminiProvider()
    except:
        try:
            provider = GroqProvider()
        except Exception as e:
            print(f"Error: {e}")
            print("Please set GOOGLE_API_KEY or GROQ_API_KEY in .env")
            return
    
    router = LLMRouter(provider)
    
    print("="*60)
    print("Customer Service Bot")
    print("="*60)
    print(f"Using: {provider.get_model_name()}")
    print("Type 'quit' to exit\n")
    
    session_id = "main-session"
    
    while True:
        query = input("\nYou: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        result = router.route(query, session_id)
        
        print(f"\nBot: {result.response}")
        print(f"[Intent: {result.intent.value}, Confidence: {result.confidence:.2f}]")


if __name__ == "__main__":
    main()