if __name__ == "__main__":
    # Входные данные
    universe = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    subsets = {
        'S1': {1, 2, 3},
        'S2': {2, 4, 6},
        'S3': {3, 5, 7},
        'S4': {1, 4, 7, 10},
        'S5': {5, 6, 8, 9}
    }
    
    print("=" * 60)
    print("ЖАДНЫЙ АЛГОРИТМ ПОКРЫТИЯ МНОЖЕСТВ")
    print("=" * 60)
    
    # Запускаем алгоритм
    result = greedy_set_cover(universe, subsets)
    
    # Детальный анализ результата
    print("\nДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТА:")
    print("-" * 40)
    
    final_coverage, coverage_details = calculate_coverage(result, subsets)
    
    print("Выбранные множества и их вклад:")
    for set_name in result:
        print(f"  {set_name}: {subsets[set_name]}")
    
    print(f"\nИтоговое покрытие: {final_coverage}")
    print(f"Исходный универсум: {universe}")
    print(f"Эффективность покрытия: {len(final_coverage)}/{len(universe)} элементов")
    
    # Проверяем, нет ли дублирующихся покрытий
    print(f"\nСТАТИСТИКА:")
    print(f"Количество выбранных множеств: {len(result)}")
    print(f"Размер универсума: {len(universe)}")
    print(f"Процент покрытия: {len(final_coverage)/len(universe)*100:.1f}%")
    
    # Проверяем, какие элементы могли быть пропущены
    missing = universe - final_coverage
    if missing:
        print(f"Пропущенные элементы: {missing}")
    else:
        print(f"Все элементы успешно покрыты!")
