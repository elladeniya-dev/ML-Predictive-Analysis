import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ── Load Models ───────────────────────────────────────────
model      = pickle.load(open('model.pkl',        'rb'))
scaler     = pickle.load(open('scaler.pkl',       'rb'))
fe_scaler  = pickle.load(open('fe_scaler.pkl',    'rb'))
feat_names = pickle.load(open('feature_names.pkl','rb'))

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title = "Hotel Booking Predictor",
    page_icon  = "🏨",
    layout     = "wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        border: 1px solid #0f3460;
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        color: #a8b2d8;
        font-size: 1rem;
        margin: 5px 0 0 0;
    }
    .section-card {
        background: #16213e;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #0f3460;
        margin-bottom: 20px;
    }
    .section-title {
        color: #e94560;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e94560;
    }
    .predict-btn {
        background: #e94560 !important;
        color: white !important;
        font-size: 1.2rem !important;
        padding: 15px !important;
        border-radius: 10px !important;
    }
    .result-cancelled {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(255,68,68,0.4);
    }
    .result-safe {
        background: linear-gradient(135deg, #00b09b, #096e4a);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,176,155,0.4);
    }
    .metric-card {
        background: #0f3460;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e94560;
    }
    .risk-high {
        background: #ff4444;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background: #ff8800;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background: #00b09b;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏨 Hotel Booking Cancellation Predictor</h1>
    <p>IT4060 — Machine Learning | 
       Member 1 — Logistic Regression</p>
    <p>Predict whether a hotel booking will be 
       cancelled or not</p>
</div>
""", unsafe_allow_html=True)

# ── Input Form ────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="section-card">
    <div class="section-title">👤 Guest Information</div>
    </div>""", unsafe_allow_html=True)

    hotel = st.selectbox(
        "Hotel Type",
        options=[0, 1],
        format_func=lambda x:
        "🏙️ City Hotel" if x==0 else "🏖️ Resort Hotel")
    adults = st.number_input(
        "Adults", min_value=1,
        max_value=10, value=2)
    children = st.number_input(
        "Children", min_value=0,
        max_value=10, value=0)
    is_repeated = st.selectbox(
        "Repeated Guest",
        options=[0, 1],
        format_func=lambda x:
        "❌ No" if x==0 else "✅ Yes")
    customer_type = st.selectbox(
        "Customer Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: [
            'Contract', 'Group',
            'Transient', 'Transient-Party'][x])

with col2:
    st.markdown("""
    <div class="section-card">
    <div class="section-title">📅 Booking Information</div>
    </div>""", unsafe_allow_html=True)

    lead_time = st.slider(
        "Lead Time (days)",
        min_value=0, max_value=500, value=90)
    weekend_nights = st.number_input(
        "Weekend Nights",
        min_value=0, max_value=20, value=2)
    week_nights = st.number_input(
        "Week Nights",
        min_value=0, max_value=30, value=5)
    booking_changes = st.number_input(
        "Booking Changes",
        min_value=0, max_value=20, value=0)
    special_req = st.number_input(
        "Special Requests",
        min_value=0, max_value=10, value=0)

with col3:
    st.markdown("""
    <div class="section-card">
    <div class="section-title">💳 Payment Information</div>
    </div>""", unsafe_allow_html=True)

    deposit_type = st.selectbox(
        "Deposit Type",
        options=[0, 1],
        format_func=lambda x:
        "✅ No Deposit" if x==0
        else "❌ Non Refundable")
    adr = st.number_input(
        "Average Daily Rate ($)",
        min_value=0.0,
        max_value=1000.0, value=100.0)
    prev_cancel = st.number_input(
        "Previous Cancellations",
        min_value=0, max_value=20, value=0)
    parking = st.number_input(
        "Car Parking Spaces",
        min_value=0, max_value=5, value=0)
    market_segment = st.selectbox(
        "Market Segment",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: [
            'Aviation', 'Complementary',
            'Corporate', 'Direct', 'Groups',
            'Offline TA', 'Online TA'][x])

st.divider()

# ── Predict Button ────────────────────────────────────────
predict = st.button(
    "🔮 PREDICT CANCELLATION",
    type="primary",
    use_container_width=True)

if predict:
    # Build input
    input_dict = {col: 0 for col in feat_names}
    input_dict['hotel']                       = hotel
    input_dict['lead_time']                   = lead_time
    input_dict['arrival_date_year']           = 2017
    input_dict['arrival_date_month']          = 8
    input_dict['arrival_date_week_number']    = 32
    input_dict['arrival_date_day_of_month']   = 15
    input_dict['stays_in_weekend_nights']     = weekend_nights
    input_dict['stays_in_week_nights']        = week_nights
    input_dict['adults']                      = adults
    input_dict['children']                    = children
    input_dict['is_repeated_guest']           = is_repeated
    input_dict['previous_cancellations']      = prev_cancel
    input_dict['booking_changes']             = booking_changes
    input_dict['deposit_type']                = deposit_type
    input_dict['adr']                         = adr
    input_dict['total_of_special_requests']   = special_req
    input_dict['required_car_parking_spaces'] = parking
    input_dict['market_segment']              = market_segment
    input_dict['customer_type']               = customer_type

    # Pipeline
    input_df     = pd.DataFrame(
                   [input_dict])[feat_names]
    input_scaled = scaler.transform(input_df)

    idx = {col: i for i, col in enumerate(feat_names)}

    f1 = (input_scaled[:,
          idx['stays_in_weekend_nights']] +
          input_scaled[:,
          idx['stays_in_week_nights']])
    f2 = (input_scaled[:,idx['lead_time']] *
          input_scaled[:,
          idx['previous_cancellations']])
    f3 = (input_scaled[:,
          idx['total_of_special_requests']] +
          input_scaled[:,idx['booking_changes']])
    f4 = (input_scaled[:,idx['lead_time']] /
         (input_scaled[:,
          idx['total_of_special_requests']]+1))

    new_feats        = np.column_stack(
                       [f1,f2,f3,f4])
    new_feats_scaled = fe_scaler.transform(new_feats)
    input_fe         = np.column_stack(
                       [input_scaled,new_feats_scaled])

    prediction  = model.predict(input_fe)[0]
    probability = model.predict_proba(input_fe)[0]

    # ── Result ────────────────────────────────────────────
    if prediction == 1:
        st.markdown(f"""
        <div class="result-cancelled">
            ❌ BOOKING WILL BE CANCELLED
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
            ✅ BOOKING WILL NOT BE CANCELLED
        </div>""", unsafe_allow_html=True)

    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Cancel Probability",
                  f"{probability[1]*100:.2f}%")
    with m2:
        st.metric("Stay Probability",
                  f"{probability[0]*100:.2f}%")
    with m3:
        confidence = max(probability) * 100
        st.metric("Confidence",
                  f"{confidence:.2f}%")

    # Risk Level
    st.divider()
    if prediction == 1:
        if probability[1] > 0.8:
            st.markdown("""
            <div class="risk-high">
                🔴 HIGH RISK — Very likely to cancel!
            </div>""", unsafe_allow_html=True)
        elif probability[1] > 0.6:
            st.markdown("""
            <div class="risk-medium">
                🟡 MEDIUM RISK — Likely to cancel!
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-medium">
                🟡 LOW-MEDIUM RISK — May cancel!
            </div>""", unsafe_allow_html=True)
    else:
        if probability[0] > 0.8:
            st.markdown("""
            <div class="risk-low">
                🟢 LOW RISK — Very likely to stay!
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-low">
                🟢 MEDIUM — Probably will stay!
            </div>""", unsafe_allow_html=True)

    # Key Factors
    st.divider()
    st.subheader("📊 Key Factors")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Lead Time",
                  f"{lead_time} days")
        st.metric("Deposit",
                  "Non-Refund" if deposit_type==1
                  else "No Deposit")
    with c2:
        st.metric("Prev Cancellations",
                  prev_cancel)
        st.metric("Special Requests",
                  special_req)
    with c3:
        st.metric("Repeated Guest",
                  "Yes" if is_repeated==1
                  else "No")
        st.metric("Daily Rate",
                  f"${adr:.2f}")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#a8b2d8;'>
    IT4060 Machine Learning | 
    Hotel Booking Cancellation Prediction |
    Logistic Regression
</div>
""", unsafe_allow_html=True)