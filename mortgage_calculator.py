import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math


st.title("Mortgage Repayments Calcuulator")

st.write("### Input Data")
col1, col2 = st.columns(2)
home_value = col1.number_input("Home value", min_value=0, value=500000)
deposit = col1.number_input("Deposit", min_value=0, value=100000)
interest_rate = col2.number_input("Interest Rate (in %)", min_value=0.0, value=0.5)
loan_term = col2.number_input("Loan term (in years)", min_value=1, value=30)

# Calcular los pagos mensuales del préstamo
loan_amount = home_value - deposit
monthly_interest_rate = (interest_rate / 100) / 12
number_of_payments = loan_term * 12

monthly_payment = (
    loan_amount
    * (monthly_interest_rate * (1 + monthly_interest_rate) ** number_of_payments)
    / ((1 + monthly_interest_rate) ** number_of_payments - 1)
)



# Display the repayments
totoal_payments = monthly_payment * number_of_payments
total_interest = totoal_payments - loan_amount

st.write("### Reppayments")
col1, col2, col3 = st.columns(3)
col1.metric(label="Monthly Repayments", value=f"${monthly_payment:,.2f}")
col2.metric(label="Total Repayments", value=f"${totoal_payments:,.0f}")
col3.metric(label="Monthly Interest", value=f"${total_interest:,.0f}")


# Crear un cronograma con el calendario de pagos.
schedule = []
remaining_balance = loan_amount

for i in range(1, number_of_payments + 1):
    interest_payment = remaining_balance * monthly_interest_rate
    monthly_payment = monthly_payment - interest_payment
    principal_payment = monthly_payment - interest_payment
    remaining_balance -= principal_payment
    year = math.ceil(i / 12)  # Calcular el año del préstamo
    schedule.append(
        [
            i,
            monthly_payment,
            principal_payment,
            interest_payment,
            remaining_balance,
            year,
        ]
    )

df = pd.DataFrame(
    schedule,
    columns=["Month", "Payment", "Principal", "Interest", "Remaining Balance", "Year"]
)

# Display the dataframe as a chart
st.write("### Payment Schedule")
payments_df = df[["Year", "Remaining Balance"]].groupby("Year").min()
st.line_chart(payments_df)