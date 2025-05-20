def calculate_sum(a: float, b: float) -> float:
    if a is None or b is None:
        raise ValueError("❌ Invalid arguments received: a or b is None")
    print(f"🧮 calculate_sum() called with a={a}, b={b}")
    return float(a) + float(b)
def get_all_function_map():
    return {
        "calculate_sum": calculate_sum,
        # add more functions here
    }