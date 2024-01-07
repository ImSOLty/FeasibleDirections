import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QTextEdit, QVBoxLayout, QHBoxLayout, QComboBox, \
    QPushButton, QFrame, QDialog, QMessageBox
from sympy import symbols, sympify, diff, Eq, solve
from scipy.optimize import linprog, minimize_scalar, minimize
import re


def parse_expression(inp: str):
    inp = inp.replace('^', '**')
    x_array = set(re.findall(r'x\d+', inp))
    return symbols(x_array), sympify(inp)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.started = False
        self.limitations = []

    def init_ui(self):
        main_layout = QVBoxLayout()

        name_font = QFont()
        name_font.setBold(True)
        name_font.setPixelSize(19)

        label_font = QFont()
        label_font.setBold(True)

        # Create labels
        name_label = QLabel('Method of Feasible Directions', self)
        objective_label = QLabel('Objective function', self)
        constraints_label = QLabel('Constraints', self)
        actions_label = QLabel('Actions', self)
        logs_label = QLabel('Logs', self)
        start_label = QLabel('Start:', self)
        error_label = QLabel('Error:', self)
        for label in [name_label, objective_label, constraints_label, actions_label, logs_label, start_label,
                      error_label]:
            label.setFont(label_font)
            label.setAlignment(Qt.AlignCenter)
        # Create division lines
        lines = [QFrame(self), QFrame(self), QFrame(self)]
        for line in lines:
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)

        # Create help and naming
        name_layout = QHBoxLayout()
        help_button = QPushButton('?')
        help_button.setFixedSize(25, 25)
        name_label.setFont(name_font)
        name_layout.addWidget(name_label)
        name_layout.addWidget(help_button)
        main_layout.addLayout(name_layout)

        target_layout = QHBoxLayout()
        self.target_input = QLineEdit()
        self.target_input.setPlaceholderText('Objective function')
        target_arrow = QLabel('→ min')
        target_arrow.setAlignment(Qt.AlignCenter)
        target_layout.addWidget(self.target_input, stretch=10)
        target_layout.addWidget(target_arrow, stretch=1)

        main_layout.addWidget(lines[0])
        main_layout.addWidget(objective_label)
        main_layout.addLayout(target_layout)

        main_layout.addWidget(constraints_label)
        self.constraints_input = []
        for _ in range(4):
            limit_layout = QHBoxLayout()

            limit_input = QLineEdit()
            limit_input.setPlaceholderText('Constraint equation (leave empty if not used)')
            limit_layout.addWidget(limit_input, stretch=10)
            limit_text = QLabel('<= 0   ')
            limit_layout.addWidget(limit_text)
            self.constraints_input.append(limit_input)

            main_layout.addLayout(limit_layout)

        additional_layout = QHBoxLayout()
        self.start_input = QLineEdit()
        self.start_input.setPlaceholderText('(x1,x2,...,xN)')
        self.error_input = QLineEdit()
        self.error_input.setText('0.1')
        additional_layout.addWidget(start_label, stretch=1)
        additional_layout.addWidget(self.start_input, stretch=1)
        additional_layout.addWidget(error_label, stretch=1)
        additional_layout.addWidget(self.error_input, stretch=1)
        main_layout.addLayout(additional_layout)

        action_reset = QPushButton('Clear')
        action_reset.clicked.connect(self.on_reset_clicked)
        action_restart = QPushButton('Restart')
        action_restart.clicked.connect(self.on_restart_clicked)
        action_run = QPushButton('Simulate')
        action_run.clicked.connect(self.on_simulation_clicked)
        action_step = QPushButton('Start/Next Iteration')
        action_step.clicked.connect(self.on_start_clicked)
        actions_layout = QHBoxLayout()
        for action in [action_reset, action_restart, action_run, action_step]:
            actions_layout.addWidget(action)

        main_layout.addWidget(lines[1])
        main_layout.addWidget(actions_label)
        main_layout.addLayout(actions_layout)
        main_layout.addWidget(lines[2])

        self.logs = QTextEdit('', self)
        main_layout.addWidget(logs_label)
        main_layout.addWidget(self.logs)

        test_layout = QHBoxLayout()
        tests = []
        for i in range(3):
            fill_test = QPushButton(f'Test data ({i + 1})')
            test_layout.addWidget(fill_test)
            tests.append(fill_test)
        tests[0].clicked.connect(self.fill_with_testdata1_clicked)
        tests[1].clicked.connect(self.fill_with_testdata2_clicked)
        tests[2].clicked.connect(self.fill_with_testdata3_clicked)
        main_layout.addLayout(test_layout)

        self.setLayout(main_layout)
        self.setWindowTitle('Method of Feasible Directions')
        self.setFixedSize(400, 800)

        self.show()

    def fill_with_testdata1_clicked(self):
        self.on_reset_clicked()
        self.target_input.setText('x1+x2')
        self.constraints_input[0].setText('x1^2-x2')
        self.constraints_input[1].setText('x2-1')
        self.start_input.setText('(-1,1)')

    def fill_with_testdata2_clicked(self):
        self.on_reset_clicked()
        self.target_input.setText('x1^2-x1*x2+2*x2^2-4*x1-6*x2')
        self.constraints_input[0].setText('x1+x2-4')
        self.constraints_input[1].setText('-x1-2*x2+2')
        self.constraints_input[2].setText('-x1')
        self.constraints_input[3].setText('-x2')
        self.start_input.setText('(3,1)')

    def fill_with_testdata3_clicked(self):
        self.on_reset_clicked()
        self.target_input.setText('x3')
        self.constraints_input[0].setText('x1^2+2*x1*x2+2*x2^2-2*x1-x2-x3-2')
        self.constraints_input[1].setText('x1^2+x2^2-x1+x2-x3-3')
        self.constraints_input[2].setText('x1^2+x1-4*x2-x3+3')
        self.start_input.setText('(1,-1,9)')

    def on_reset_clicked(self):
        self.target_input.setText('')
        self.start_input.setText('')
        self.error_input.setText('0.1')
        for inp in self.constraints_input:
            inp.setText('')
        self.logs.clear()
        self.log('Everything cleared!\n')

    def parse_everything(self):
        self.log('Parsing data...')
        self.limitations = []
        try:
            if self.target_input.text() == '':
                self.error_box('Objective function is empty!')
                return False
            self.x_array, self.target = parse_expression(self.target_input.text())
            self.log(f'Minimize: {self.target}')
            self.target *= -1
            for inp in self.constraints_input:
                limit_input = inp.text()
                if limit_input == '':
                    continue
                limit_parsed = parse_expression(limit_input)
                self.x_array |= limit_parsed[0]
                self.limitations.append(limit_parsed[1])
            if len(self.limitations) == 0:
                self.error_box('No constraints found!')
                return False
            self.log(f'Constraints:')
            self.log('\n'.join([str(limit) + '<=0' for limit in self.limitations]))

            self.x_array = sorted(list(self.x_array), key=lambda s: s.name)
            self.log('Vars: ' + ', '.join([x.name for x in self.x_array]))
            self.x0_input = eval(self.start_input.text())
            self.log(f'Start: {self.x0_input}')
            self.error = float(self.error_input.text())
            self.log(f'Error value: {self.error}')

            if any([limit.subs([(var, num) for var, num in zip(self.x_array, self.x0_input)]) > 0 for limit in
                    self.limitations]):
                self.error_box("Check your constraints. Some of them are not being fulfilled for the set start point.")
                return False

        except SyntaxError:
            self.error_box("Can't parse some of the expressions. Make sure that provided objective and constraints "
                           "functions are valid.")
            return False
        except BaseException as e:
            self.error_box(f'Unexpected error: {e}')
            return False
        return True

    def on_simulation_clicked(self):
        if not self.started:
            if not self.parse_everything():
                return
            self.log('Simulation started!')
            self.subs = [(var, num) for var, num in zip(self.x_array, self.x0_input)]
            self.prev = None
            self.k = 0
        while True:
            self.log(f"{'-' * 30}Iteration {self.k + 1}{'-' * 30}")
            feedback, result = self.run_iteration(self.k)
            if feedback:
                break
            if self.k > 1000:
                self.error_box("Simulation execution taken too many iterations")
                return
            self.k += 1
        self.log(f"{'-' * 26}Simulation ended!{'-' * 26}")
        if result is None:
            self.log(f"Simulation stopped unexpectedly")
        else:
            self.log(f"x*={result}\nf(x*)={-self.target.subs(result)}\n")
        self.started = False

    def on_start_clicked(self):
        if self.started:
            self.log(f"{'-' * 30}Iteration {self.k + 1}{'-' * 30}")
            feedback, result = self.run_iteration(self.k)
            self.log(f"{'-' * 26}Simulation ended!{'-' * 26}")
            if result is None:
                self.log(f"Simulation stopped unexpectedly")
            else:
                self.log(f"x*={result}\nf(x*)={-self.target.subs(result)}\n")
            self.started = False
            self.k += 1
        else:
            if not self.parse_everything():
                return
            self.started = True
            self.k = 0
            self.subs = [(var, num) for var, num in zip(self.x_array, self.x0_input)]
            self.prev = None

    def on_restart_clicked(self):
        self.log("Returned back to start\n")
        self.started = False

    def error_box(self, s: str):
        self.log(f'******Error: {s}')
        QMessageBox.critical(self, 'Error!', s, QMessageBox.Ok)

    def run_iteration(self, k):
        try:
            # Step 1
            nabla_f_res = [diff(self.target, x).subs(self.subs) for x in self.x_array]
            limitations_res = [limit.subs(self.subs) for limit in self.limitations]
            self.log(f"Gradient vector: ∇f(x{k})={nabla_f_res}")

            # Step 2
            limitations_indexes = []
            for i, limit_res in enumerate(limitations_res):
                self.log(f'\tg{i}(x{k})={limit_res}')
                if abs(limit_res) <= self.error:
                    limitations_indexes.append(i)
            self.log(f"Limitation indexes: {limitations_indexes}")

            # Step 3
            if len(limitations_indexes) == 0:
                # Step 5
                s_k = nabla_f_res
                omega_k = sum(k ** 2 for k in nabla_f_res) ** (1 / 2)
            else:
                # Step 4
                omega = symbols('omg')
                s = symbols(' '.join([f's{i + 1}' for i in range(len(self.x_array))]))

                # target for inner
                c = [float(-omega.coeff(omega))] + [0.0] * len(s)  # -omega because linprog minimizes

                nabla_f = [diff(self.target, x) for x in self.x_array]
                extra_limit_expr = -sum(nabla_i * s_i for nabla_i, s_i in zip(nabla_f, s)) + omega
                limitations_inner = [extra_limit_expr]
                for index in limitations_indexes:
                    nabla_g = [diff(self.limitations[index], x) for x in self.x_array]
                    limitations_inner.append(
                        self.limitations[index] + sum(nabla_i * s_i for nabla_i, s_i in zip(nabla_g, s)) + omega)
                limitations_inner = list(map(lambda x: x.subs(self.subs), limitations_inner))

                # left part of limitations
                A = [
                    [float(expr.coeff(x)) for x in [omega] + list(s)]
                    for expr in limitations_inner
                ]
                b = [0.0] * len(limitations_inner)

                # Решение задачи линейного программирования
                result = linprog(c, A_ub=A, b_ub=b,
                                 bounds=[(float('-inf'), None)] + [(-1, 1) for _ in range(len(c) - 1)])
                # print(f'Left part inner: {A}')
                # print(f'Right part inner: {b}')

                omega_k = result.x[0]
                s_k = result.x[1:]

            self.log(f"\tσk: {omega_k}")
            self.log(f"\tSk: {s_k}")
            # Step 6
            if abs(omega_k) < self.error:
                self.log('Found result!')
                return True, self.subs
            self.prev = self.target.subs(self.subs)

            # Step 7
            # Step 7.1.
            beta = symbols('beta')
            betas_l = []
            for limit in self.limitations:
                equation = Eq(limit.subs([(sub[0], sub[1] + beta * s_k[i]) for i, sub in enumerate(self.subs)]), 0)
                solution = list(map(float, solve(equation, beta)))
                betas_l.extend(filter(lambda x: x > 0, solution))
            beta_star = min(betas_l)
            self.log(f"\tBeta*: {beta_star}")

            # Step 7.2

            def objective_function(beta):
                return -float(self.target.subs([(sub[0], sub[1] + beta * s_k[i]) for i, sub in enumerate(self.subs)]))

            # Find the maximum value and the corresponding x
            result = minimize_scalar(objective_function, bounds=(0.0, beta_star), method='bounded')

            self.subs = [(sub[0], sub[1] + result.x * s_k[i]) for i, sub in enumerate(self.subs)]
            self.log(f'Current: {self.subs}')

            return False, []
        except BaseException as e:
            self.error_box(f"Unexpected error: {e}")
            return True, None

    def log(self, s: str):
        self.logs.append(s)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
