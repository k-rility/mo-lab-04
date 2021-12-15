import numpy as np
import matplotlib.pyplot as plt


# метод замены критериев ограничениями
class MethodReplaceCriteria:
    def __init__(self, matrix, criteria, index):
        self.matrix = matrix.copy()
        self.criteria = criteria.copy()
        self.index = index
        self.matrix_normalization()
        self.ans = self.solution()

    def matrix_normalization(self):
        min_ = np.min(self.matrix, axis=0)
        max_ = np.max(self.matrix, axis=0)
        for i in range(4):
            if i != self.index:
                for j in range(4):
                    self.matrix[j][i] = (self.matrix[j][i] - min_[i]) / (max_[i] - min_[i])
        print(f'нормированная матрица А:\n {self.matrix}')

    def check_lower_threshold(self, row):
        for i in range(4):
            if i != self.index:
                if row[i] < self.criteria[i]:
                    return False
        return True

    def solution(self):
        ans_ = None
        for j in range(4):
            if self.check_lower_threshold(self.matrix[j]):
                if ans_ is None or self.matrix[ans_][self.index] < self.matrix[j][self.index]:
                    ans_ = j
        return ans_


# Формирование и сужение множества Парето
# Выберем в качестве двух основных критериев: качество лечения и качество питания

class SetPareto:
    def __init__(self, matrix, criteria1, criteria3):
        self.x = matrix[:, criteria1][:]
        self.y = matrix[:, criteria3][:]
        self.max_x = np.max(self.x)
        self.max_y = np.max(self.y)
        self.answer = self.solution_set_pareto()

    def graph_set_pareto(self):
        plt.plot(self.x, self.y, 'o')
        plt.show()

    def distance(self, x, y):
        return np.max([np.abs(x - self.max_x), np.abs(y - self.max_y)])

    def solution_set_pareto(self):
        min_distance = self.distance(0., 0.)
        answer = None
        for i in range(4):
            dis = self.distance(self.x[i], self.y[i])
            if dis < min_distance:
                min_distance = dis
                answer = i

        return answer


# Взвешивание и объединение критериев
class MethodWeighingCombiningCriteria:
    def __init__(self, vector_criteria, matrix):
        self.matrix = matrix.copy()
        self.vector_criteria = vector_criteria.copy()
        self.criteria_weights = self.method_pairwise_comparison()
        self.matrix_normalization()
        self.answer = self.get_solution()

    @staticmethod
    def compare_criteria(criteria1, criteria2):
        if criteria1 > criteria2:
            return 1
        elif criteria1 == criteria2:
            return 0.5
        return 0

    def method_pairwise_comparison(self):

        expert_grade = np.array([
            [self.compare_criteria(criteria_1, criteria_2) for criteria_2 in
             self.vector_criteria]
            for criteria_1 in self.vector_criteria
        ])

        print(f"Экспертные оценки:\n {expert_grade}")
        criteria_weights = np.array([sum(expert_grade[i]) - 0.5 for i in
                                     range(4)])
        print(f"Вектор весов критериев: {criteria_weights}")
        criteria_weights = np.divide(criteria_weights, sum(criteria_weights))
        print(f"Нормированный вектор весов критериев: {criteria_weights}")
        return criteria_weights

    def matrix_normalization(self):

        for col in range(len(self.matrix)):
            self.matrix[:, col] = np.divide(self.matrix[:, col],
                                            np.sum(self.matrix[:, col]))

        print(f"Нормированная матрица A:\n {self.matrix}")

    def get_solution(self):

        self.criteria_weights = np.transpose(np.matrix(self.vector_criteria))
        self.matrix = np.matrix(self.matrix)
        solutions = self.matrix * self.criteria_weights
        max_result = 0
        answer = None
        for i in range(4):
            if solutions[i] > max_result:
                max_result = solutions[i]
        answer = i
        return answer

    def compare(self, rating1, rating2):
        delta = rating1 - rating2
        if delta == 3:
            return 7
        elif delta == 2:
            return 5
        elif delta == 1:
            return 3
        elif delta == 0:
            return 1
        elif delta == -1:
            return 1 / 3
        elif delta == -2:
            return 1 / 5
        else:
            return 1 / 7
        
class MethodHierarchyAnalysis:
    def __init__(self, matrix, vector_priorities):
        self.__matrix = matrix.copy()
        self.__vector_priorities = vector_priorities

        self.matrix_criteria0 = self.pairwise_comparison(0)
        self.matrix_criteria1 = self.pairwise_comparison(1)
        self.matrix_criteria2 = self.pairwise_comparison(2)
        self.matrix_criteria3 = self.pairwise_comparison(3)
        self.matrix_criteria = np.transpose(
            np.matrix([self.matrix_criteria0, self.matrix_criteria1, self.matrix_criteria2, self.matrix_criteria3]))
        self.matrix_priorities = np.transpose(np.matrix(self.comparison_priorities()))
        self.answer = self.get_solution()

    def pairwise_comparison(self, criteria_index):
        new_matrix = np.array(
            [[self.compare(self.__matrix[row_1][criteria_index], self.__matrix[row_2][criteria_index]) for
              row_2 in range(len(self.__matrix))] for row_1 in range(len(self.__matrix))])
        sum_line = np.array([np.sum(new_matrix[i]) for i in range(len(new_matrix))])
        sum_line = np.divide(sum_line, np.sum(sum_line))
        return sum_line

    def comparison_priorities(self):
        new_matrix = np.array(
            [[self.compare(self.__vector_priorities[row_1], self.__vector_priorities[row_2]) for row_2 in
              range(len(self.__matrix))] for row_1 in range(len(self.__matrix))])
        sum_line = np.array([np.sum(new_matrix[i]) for i in range(len(new_matrix))])
        sum_line = np.divide(sum_line, np.sum(sum_line))
        return sum_line

    def get_solution(self):
        solutions = self.matrix_criteria * self.matrix_priorities
        max_result = 0
        answer = None
        for i in range(len(self.__matrix)):
            if solutions[i] > max_result:
                max_result = solutions[i]
                answer = i
        return answer
