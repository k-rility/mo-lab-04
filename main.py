from Methods import *

# Альтернативы:
# А. "Липецк", г. Липецк.
# B. "Сосновый бор", Тамбовский район.
# С. "Лесная жемчужина", г. Котовск.
# D. "Сосны", г. Пенза

# Критерии:
# 1. Качество лечения.
# 2. Уровень сервиса.
# 3. Качество питания.
# 4. Расстояние от Москвы

# вектор весов критериев
vector_criteria = [8, 4, 6, 2]
vector_criteria = np.divide(vector_criteria, sum(vector_criteria))
print(f'вектор весов критериев: {vector_criteria}')

A = np.array([
    [7., 3., 3., 4.],
    [2., 6., 3., 5.],
    [4., 3., 6., 5.],
    [6., 5., 7., 2.]
])

# матрица А оценок для альтернатив
print(f'матрица А оценок для альтернатив: {vector_criteria}')

# выберем в качестве главного критерия качество лечения
# установим минимально допустимые уровни для остальных критериев
criteria__ = np.array([None, 0.3, 0.6, 0.1])
print(f'допустимые значения: {criteria__}')

vec_prior = np.array([3, 1, 2, 0])

if __name__ == '__main__':
    print("Метод № 1\n")

    method1 = MethodReplaceCriteria(matrix=A, criteria=criteria__, index=0)
    if method1.ans is None:
        print(f'подходящего решения нет')
    else:
        print(f'самое лучшее решение: {method1.ans}')

    print('-----------------------------------------------------------')
    print("\nМетод № 2\n")

    method2 = SetPareto(A, 0, 2)
    method2.graph_set_pareto()
    print(f"Минимальное расстояние до точки {method2.answer}, следовательно решение {method2.answer} оптимально")

    print('-----------------------------------------------------------')
    print("\nМетод № 3\n")

    method3 = MethodWeighingCombiningCriteria(matrix=A,
                                              vector_criteria=vector_criteria)
    print(f"Значения объединенного критерия для всех альтернатив:\n {method3.matrix * method3.criteria_weights}")
    print(f"Наиболее приемлемой является альтернатива {method3.answer}")

    print('-----------------------------------------------------------')
    print("\nМетод № 4\n")

    method4 = MethodHierarchyAnalysis(matrix=A,
                                      vector_priorities=vec_prior)

    print(method4.matrix_priorities)

    print(f"Наиболее приемлемой является альтернатива {method4.answer}")
