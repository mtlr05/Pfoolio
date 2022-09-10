import xlwings as xw
import Pfoolio as Pf


def main():
    # wb = xw.Book.caller()
    # sheet = wb.sheets[0]
    # if sheet["A1"].value == "Hello xlwings!":
        # sheet["A1"].value = "Bye xlwings!"
    # else:
        # sheet["A1"].value = "Hello xlwings!"
    Pf.nbapf_from_xl()
    Pf.opti_to_xl()
    Pf.fig_to_xl()

@xw.func
def hello(name):
    return f"Hello {name}!"


if __name__ == "__main__":
    xw.Book("PfUI.xlsm").set_mock_caller()
    main()
