import pandas as pd
import statistics
import matplotlib.pyplot as plt

def u_execute_irctc_stock_analysis():
    """
    LOADS IRCTC STOCK PRICE DATA, PERFORMS STATISTICAL ANALYSIS,
    PROBABILITY CALCULATION, AND VISUALIZATION FOR TASK A3.
    """

    # LOAD EXCEL DATA FROM THE "IRCTC Stock Price" SHEET
    u_stock_data = pd.read_excel(
        r"C:\Users\Udhaya\sem5_ML\Lab Session Data.xlsx",
        sheet_name="IRCTC Stock Price"
    )

    # CLEAN COLUMN NAMES TO REMOVE EXTRA SPACES
    u_stock_data.columns = u_stock_data.columns.str.strip()

    # CONVERT 'Chg%' TO FLOAT i.e. REMOVE '%' AND CONVERT TO NUMBER
    u_stock_data['Chg%'] = u_stock_data['Chg%'].astype(str).str.replace('%', '').astype(float)

    # CALCULATE POPULATION MEAN AND VARIANCE FOR PRICE
    u_price_list = u_stock_data['Price'].tolist()
    u_price_mean = statistics.mean(u_price_list)
    u_price_variance = statistics.variance(u_price_list)

    # CALCULATE MEAN PRICE ON WEDNESDAYS
    u_wed_prices = u_stock_data[u_stock_data['Day'] == 'Wed']['Price']
    u_wed_mean = statistics.mean(u_wed_prices)

    # CALCULATE MEAN PRICE IN APRIL (IF APRIL DATA EXISTS)
    u_april_prices = u_stock_data[u_stock_data['Month'] == 'Apr']['Price']
    u_april_mean = statistics.mean(u_april_prices) if not u_april_prices.empty else None

    # CALCULATE PROBABILITY OF LOSS (Chg% < 0)
    u_loss_probability = (u_stock_data['Chg%'] < 0).sum() / len(u_stock_data)

    # CALCULATE PROBABILITY OF PROFIT ON WEDNESDAY
    u_wednesday_data = u_stock_data[u_stock_data['Day'] == 'Wed']
    u_profit_on_wed_prob = (u_wednesday_data['Chg%'] > 0).sum() / len(u_wednesday_data)

    # CONDITIONAL PROBABILITY OF PROFIT GIVEN WEDNESDAY (SAME VALUE)
    u_conditional_profit_given_wed = u_profit_on_wed_prob

    # PLOT SCATTER OF Chg% AGAINST DAY OF WEEK
    plt.figure(figsize=(8, 5))
    plt.scatter(u_stock_data['Day'], u_stock_data['Chg%'], color='orange')
    plt.title("Chg% vs Day of the Week")
    plt.xlabel("Day")
    plt.ylabel("Chg%")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # DISPLAY RESULTS
    print(" POPULATION MEAN OF PRICE:", round(u_price_mean, 2))
    print(" POPULATION VARIANCE OF PRICE:", round(u_price_variance, 2))
    print(" MEAN PRICE ON WEDNESDAYS:", round(u_wed_mean, 2))
    print(" MEAN PRICE IN APRIL:", round(u_april_mean, 2) if u_april_mean else "No April data available")
    print(" PROBABILITY OF MAKING A LOSS:", round(u_loss_probability, 3))
    print(" PROBABILITY OF PROFIT ON WEDNESDAY:", round(u_profit_on_wed_prob, 3))
    print(" CONDITIONAL PROBABILITY OF PROFIT GIVEN WEDNESDAY:", round(u_conditional_profit_given_wed, 3))

# RUN THE FUNCTION
u_execute_irctc_stock_analysis()
