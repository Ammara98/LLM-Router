import time
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from providers import GeminiProvider, GroqProvider, OllamaProvider
from router import LLMRouter
from dtos import Intent

load_dotenv()


def load_test_cases():
    test_file = Path("data/test_cases.json")
    
    if not test_file.exists():
        print(f"Test file not found at: {test_file.absolute()}")
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    test_cases = []
    for case in data['test_cases']:
        query = case['query']
        intent_str = case['expected_intent']

        if intent_str == "faq":
            intent = Intent.FAQ
        elif intent_str == "order_status":
            intent = Intent.ORDER_STATUS
        elif intent_str == "unclear":
            intent = Intent.UNCLEAR
        else:
            print(f"Unknown intent '{intent_str}' for query: {query}")
            continue
        
        test_cases.append((query, intent))
    
    print(f"Loaded {len(test_cases)} test cases from {test_file}")
    return test_cases


TEST_CASES = load_test_cases()


def evaluate_model(provider_name: str, provider):
    print(f"\n{'='*60}")
    print(f"Testing: {provider_name}")
    print(f"{'='*60}")
    
    router = LLMRouter(provider)
    correct = 0
    total = len(TEST_CASES)
    latencies = []
    
    intent_stats = {
        "faq": {"correct": 0, "total": 0},
        "order_status": {"correct": 0, "total": 0},
        "unclear": {"correct": 0, "total": 0}
    }
    
    for i, (query, expected) in enumerate(TEST_CASES, 1):
        try:
            start = time.time()
            result = router.route(query)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            is_correct = result.intent == expected
            if is_correct:
                correct += 1
            
            intent_key = expected.value
            intent_stats[intent_key]["total"] += 1
            if is_correct:
                intent_stats[intent_key]["correct"] += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} [{i}/{total}] {query[:40]:<40} Expected: {expected.value:<15} Got: {result.intent.value:<15} ({latency:.0f}ms)", flush=True)
        except Exception as e:
            print(f"✗ [{i}/{total}] ERROR: {query[:40]:<40} {str(e)[:50]}", flush=True)
            intent_key = expected.value
            intent_stats[intent_key]["total"] += 1
    
    accuracy = correct / total * 100
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    print(f"\n{'─'*60}")
    print(f"Results:")
    print(f"  Overall Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")
    print(f"  Avg Latency:      {avg_latency:.0f}ms")
    print(f"  Total Time:       {sum(latencies)/1000:.1f}s")
    
    # Print breakdown by intent
    print(f"\n  Breakdown by Intent:")
    for intent_name, stats in intent_stats.items():
        if stats["total"] > 0:
            intent_acc = (stats["correct"] / stats["total"]) * 100
            print(f"    {intent_name.upper():<15} {intent_acc:>5.1f}% ({stats['correct']}/{stats['total']})")
    
    return {
        "model": provider_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "latency": avg_latency,
        "total_time": sum(latencies)/1000,
        "intent_breakdown": intent_stats
    }


def main():
    print("="*60)
    print("MODEL EVALUATION - Testing All Providers")
    print("="*60)
    
    results = []
    providers_to_test = []
    
    if os.getenv("GOOGLE_API_KEY"):
        providers_to_test.append(("Gemini Flash", lambda: GeminiProvider()))
    
    if os.getenv("GROQ_API_KEY"):
        providers_to_test.append(("Groq Llama", lambda: GroqProvider()))
    
    # if os.getenv("HUGGINGFACE_API_KEY"):
    #     providers_to_test.append(("HF Mistral", lambda: HuggingFaceProvider()))
    
    ollama_models = [
        ("Ollama Qwen2.5 3B", lambda: OllamaProvider(model_name="qwen2.5:3b")),
        ("Ollama Gemma2 9B", lambda: OllamaProvider(model_name="gemma2:9b")),
    ]
    
    try:
        test_provider = OllamaProvider(model_name="qwen2.5:3b")
        _ = test_provider.get_model_name()
        providers_to_test.extend(ollama_models)
        print("✓ Ollama server detected")
        del test_provider  # Clean up test provider
    except Exception as e:
        print(f"\n Ollama not available: {e}")
    
    if not providers_to_test:
        print("No API keys or Ollama found!")
        return
    
    print(f"\n Found {len(providers_to_test)} provider(s) to test\n")
    
    # Test each provider
    for name, provider_factory in providers_to_test:
        try:
            provider = provider_factory()
            result = evaluate_model(name, provider)
            results.append(result)
            time.sleep(1)
        except Exception as e:
            print(f"\n {name} failed: {e}\n")
    
    # Print comparison
    if results:
        print(f"\n{'='*60}")
        print("COMPARISON TABLE")
        print(f"{'='*60}")
        print(f"{'Model':<25} {'Accuracy':<12} {'Latency':<12} {'Total Time'}")
        print("─"*60)
        
        for r in sorted(results, key=lambda x: (-x['accuracy'], x['latency'])):
            print(f"{r['model']:<25} {r['accuracy']:>6.1f}% {r['latency']:>8.0f}ms {r['total_time']:>8.1f}s")
        
        # Recommendations
        best = max(results, key=lambda x: x['accuracy'])
        fastest = min(results, key=lambda x: x['latency'])
        
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"Best Accuracy:  {best['model']} ({best['accuracy']:.1f}%)")
        print(f"Fastest:        {fastest['model']} ({fastest['latency']:.0f}ms avg)")
        
        # Determine overall best
        if best['model'] == fastest['model']:
            print(f"\n Overall Best: {best['model']}")
            print(f"   Reason: Best accuracy AND fastest response time")
        elif best['accuracy'] >= 90:
            print(f"\n Recommended: {best['model']}")
            print(f"   Reason: Highest routing accuracy ({best['accuracy']:.1f}%)")
        else:
            print(f"\n Recommended: Consider {fastest['model']} if speed matters")
            print(f"   Or use {best['model']} for better accuracy")
        
        # Ollama-specific recommendation
        ollama_results = [r for r in results if r['model'].startswith('Ollama')]
        if ollama_results:
            best_ollama = max(ollama_results, key=lambda x: x['accuracy'])
            print(f"\n  Best Local Model (Ollama): {best_ollama['model']}")
            print(f"   Accuracy: {best_ollama['accuracy']:.1f}% | Latency: {best_ollama['latency']:.0f}ms")
            print(f"   Benefit: Free, private, no API keys needed")
        
        import json
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n Results saved to evaluation_results.json")


if __name__ == "__main__":
    main()