import plotguy

all_para_combination = []
number_of_core = 4


def run_plotguy(all_para_combination, number_of_core=4):
    plotguy.generate_backtest_result(
        all_para_combination=all_para_combination,
        number_of_core=number_of_core
    )

    app = plotguy.plot(
        mode='equity_curves',
        all_para_combination=all_para_combination
    )

    app.run_server(port=8902)
