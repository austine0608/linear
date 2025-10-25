import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------
st.set_page_config(page_title="USA Housing Linear Regression", page_icon="🏠", layout="wide")

st.title("🏠 USA Housing Price Prediction using Linear Regression")
st.write("""
Explore the **USA Housing dataset**, train a **Linear Regression model**, 
and make interactive predictions — all in one Streamlit app with Plotly visualizations.
""")

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------
st.subheader("📂 Load Dataset")

df = pd.read_csv('USA_Housing.csv')

with st.expander('Load Dataset'):
    st.dataframe(df)

# -----------------------------------------------------
# Data Summary and Visualization
# -----------------------------------------------------
st.subheader("📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("### Summary Statistics")
    st.dataframe(df.describe())

with col2:
    st.write("### Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                         color_continuous_scale="RdBu_r", title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------------------------------
# Feature Distributions
# -----------------------------------------------------
st.write("### Feature Distributions (Interactive)")

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
selected_feature = st.selectbox("Select feature to visualize:", numeric_cols)

fig_hist = px.histogram(df, x=selected_feature, nbins=40, 
                        title=f"Distribution of {selected_feature}", 
                        color_discrete_sequence=['teal'], 
                        marginal="box", 
                        hover_data=df.columns)
st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------------------------------
# Feature Selection
# -----------------------------------------------------
st.subheader("⚙️ Model Setup")

all_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

x_features = st.multiselect(
    "Select features (X)",
    all_columns[:-1],
    default=[
        'Avg. Area Income',
        'Avg. Area House Age',
        'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms',
        'Area Population'
    ]
)

y_feature = st.selectbox("Select target variable (y)", all_columns, index=len(all_columns)-1)

# -----------------------------------------------------
# Model Training
# -----------------------------------------------------
if len(x_features) > 0:
    X = df[x_features]
    y = df[y_feature]

    test_size = st.slider("Test size (proportion of data for testing)", 0.1, 0.5, 0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -----------------------------------------------------
    # Model Performance
    # -----------------------------------------------------
    st.subheader("📈 Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_pred):,.2f}")
    col2.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):,.2f}")
    col3.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")

    # -----------------------------------------------------
    # Coefficients
    # -----------------------------------------------------
    st.subheader("🧮 Model Coefficients")
    coef_df = pd.DataFrame({
        "Feature": x_features,
        "Coefficient": model.coef_
    })
    st.dataframe(coef_df)
    st.write(f"**Intercept:** {model.intercept_:,.2f}")

    # -----------------------------------------------------
    # Visualization: Actual vs Predicted
    # -----------------------------------------------------
    st.subheader("🎨 Actual vs Predicted Prices (Interactive)")
    fig_scatter = px.scatter(
        x=y_test,
        y=y_pred,
        labels={"x": "Actual Prices", "y": "Predicted Prices"},
        title="Actual vs Predicted Prices",
        color_discrete_sequence=['orange']
    )
    fig_scatter.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal Fit', line=dict(color='blue', dash='dot')))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # -----------------------------------------------------
    # Prediction Section
    # -----------------------------------------------------
    st.subheader("🔮 Make a New Prediction")

    input_data = {}
    for feature in x_features:
        input_data[feature] = st.number_input(f"Enter {feature}", value=float(X[feature].mean()))

    if st.button("Predict House Price"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"🏡 Predicted House Price: **${prediction:,.2f}**")

else:
    st.warning("⚠️ Please select at least one feature to continue.")



