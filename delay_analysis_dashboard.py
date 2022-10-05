import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
import math

def get_checkout_state(row):
    state = 'Unknown'
    if row['state'] == 'ended':
        if row['delay_at_checkout_in_minutes'] <= 0:
            state = "On time checkout"
        elif row['delay_at_checkout_in_minutes'] > 0:
            state = "Late checkout"
    if row['state'] == 'canceled':
        state = "Canceled"
    return state

def keep_only_ended_state(dataframe):
    condition1 = dataframe['state'] == 'On time checkout'
    condition2 = dataframe['state'] == 'Late checkout'
    return dataframe[condition1 | condition2]

def get_previous_rental_delay(row, dataframe):
    delay = np.nan
    if not math.isnan(row['previous_ended_rental_id']):
        delay = dataframe[dataframe['rental_id'] == row['previous_ended_rental_id']]['delay_at_checkout_in_minutes'].values[0]
    return delay

def get_impact_of_previous_rental_delay(row):
    impact = 'No previous rental filled out'
    if not math.isnan(row['checkin_delay_in_minutes']):
        if row['checkin_delay_in_minutes'] > 0:
            if row['state'] == 'Canceled':
                impact = 'Cancelation'
            else:
                impact = 'Late checkin'
        else:
            impact = 'No impact'
    return impact

def detect_outliers(dataframe, feature_name):
    q1 = dataframe[feature_name].quantile(0.25)
    q3 = dataframe[feature_name].quantile(0.75)
    interquartile_range = q3 - q1
    upper_whisker = math.ceil(q3 + 1.5 * interquartile_range)
    n_rows_before_outliers_drop = len(dataframe)
    mask = (dataframe[feature_name] <= upper_whisker) | (dataframe[feature_name].isna())
    nrows_after_outliers_drop = len(dataframe[mask])
    n_rows_dropped = n_rows_before_outliers_drop - nrows_after_outliers_drop
    percent_dropped = round(n_rows_dropped / n_rows_before_outliers_drop * 100)
    output = {
        'upper_whisker' : upper_whisker,
        'n_rows_dropped' : n_rows_dropped,
        'percent_dropped' : percent_dropped
    }
    return output

def show_checkout_delays_overview(dataframe, late_dataframe, upper_whisker):
    col_state_pie, col_delays_boxplot, delay_info = st.columns(3)
    with col_state_pie:
        state_pie = px.pie(
            dataframe, names = "state", color = "state", 
            height = 500, 
            color_discrete_map={
                'On time checkout':'Green', 
                'Late checkout':'Red', 
                'Canceled':'Grey'
                },
            category_orders={"state" : ['On time checkout', 'Late checkout', 'Canceled', 'Unknown']},
            title = "Rental state")
        st.plotly_chart(state_pie, use_container_width=True)
    with col_delays_boxplot:
        delays_boxplot = px.box(
            late_dataframe, y = 'delay_at_checkout_in_minutes', height = 500,
            labels = {'delay_at_checkout_in_minutes' : 'Checkout delay (minutes)'},
            range_y = [0, upper_whisker + 1],
            title = "Checkout delays breakdown (outliers hidden)")
        st.plotly_chart(delays_boxplot, use_container_width=True)
    with delay_info:
        for i in range(8):
            st.text("")
        st.metric(
            label = "", 
            value=f"{round(len(late_dataframe[late_dataframe['delay_at_checkout_in_minutes'] >= 60])/len(late_dataframe)*100)}% of late checkouts", 
            delta = "have a delay of at least 1 hour",
            delta_color = 'inverse'
            )
        st.write()
        for i in range(1):
            st.text("")
        st.metric(
            label = "", 
            value=f"{round(len(late_dataframe[late_dataframe['delay_at_checkout_in_minutes'] <= 30])/len(late_dataframe)*100)}% of late checkouts",
            delta = "have a delay of less than 30 minutes",
            delta_color = 'normal'
        )

def show_impacts_on_checkins_overview(dataframe, late_dataframe, upper_whisker):
    col_impacts_pie, col_checkin_delays_boxplot, checkin_delay_metrics = st.columns(3)
    with col_impacts_pie:
        impacts_pie = px.pie(
            dataframe, names = "impact_of_previous_rental_delay", color = "impact_of_previous_rental_delay", 
            height = 500, 
            color_discrete_map={
                'No impact':'Green', 
                'Late checkin':'Orange', 
                'Cancelation':'Red',
                'No previous rental filled out':'Grey'
                },
            category_orders={"impact_of_previous_rental_delay" : ['No impact', 'Late checkin', 'Cancelation', 'No previous rental filled out']},
            title = "Impacts on state")
        st.plotly_chart(impacts_pie, use_container_width=True)
    with col_checkin_delays_boxplot:
        checkin_delays_boxplot = px.box(
            late_dataframe, y = 'checkin_delay_in_minutes', height = 500, 
            labels = {'checkin_delay_in_minutes' : 'Checkin delay (minutes)'},
            range_y = [0, upper_whisker + 1],
            title = "Checkin delays breakdown (outliers hidden)")
        st.plotly_chart(checkin_delays_boxplot, use_container_width=True)
    with checkin_delay_metrics:
        nb_late_checkins_cancelled = len(late_dataframe[late_dataframe['impact_of_previous_rental_delay'] == 'Cancelation'])
        nb_total_canceled = len(dataframe[dataframe['state'] == 'Canceled'])
        for i in range(8):
            st.text("")
        st.metric(
            label = "", 
            value=f"{round(nb_late_checkins_cancelled / len(late_dataframe) * 100)}% of late checkins", 
            delta = "are cancelled",
            delta_color = 'inverse'
            )
        st.write()
        for i in range(1):
            st.text("")
        st.metric(
            label = "", 
            value=f"{round(nb_late_checkins_cancelled / nb_total_canceled * 100)}% of all cancelations",
            delta = "follow a late checkin",
            delta_color = 'inverse'
        )

def show_simulation_results(dataframe_before, dataframe_after, whole_df_before, whole_df_after, scope):
    col_impacts_pie, gap, simulation_metrics = st.columns([2, 1, 2])
    with col_impacts_pie:
        impacts_pie = px.pie(
            dataframe_after, names = "impact_of_previous_rental_delay", color = "impact_of_previous_rental_delay", 
            height = 500, 
            color_discrete_map={
                'No impact':'Green', 
                'Late checkin':'Orange', 
                'Cancelation':'Red',
                'No previous rental filled out':'Grey'
                },
            category_orders={"impact_of_previous_rental_delay" : ['No impact', 'Late checkin', 'Cancelation', 'No previous rental filled out']},
            title = "Impacts on state")
        st.plotly_chart(impacts_pie, use_container_width=True)
    with simulation_metrics:
        if scope == "'Connect'":
            connect_df_before = whole_df_before[whole_df_before['checkin_type']=='Connect']
            connect_df_after = whole_df_before[whole_df_before['checkin_type']=='Connect']
            nb_ended_rentals_removed = len(keep_only_ended_state(connect_df_before)) - len(keep_only_ended_state(connect_df_after))
        else:
            nb_ended_rentals_removed = len(keep_only_ended_state(whole_df_before)) - len(keep_only_ended_state(whole_df_after))
        percent_ended_rentals_removed = nb_ended_rentals_removed / len(keep_only_ended_state(whole_df_before)) * 100
        if scope == "'Connect'":
            connect_dataframe_before = dataframe_before[dataframe_before['checkin_type']=='Connect']
            connect_dataframe_after = dataframe_after[dataframe_after['checkin_type']=='Connect']
            nb_ended_rentals_removed = len(keep_only_ended_state(connect_dataframe_before)) - len(keep_only_ended_state(connect_dataframe_after))
            nb_cancelations_due_to_previous_delay_before = len(connect_df_before[dataframe_before['impact_of_previous_rental_delay'] == 'Cancelation'])
        nb_cancelations_due_to_previous_delay_after = len(dataframe_after[dataframe_after['impact_of_previous_rental_delay'] == 'Cancelation'])
        nb_cancelations_avoided = nb_cancelations_due_to_previous_delay_before - nb_cancelations_due_to_previous_delay_after
        percent_cancelations_avoided = nb_cancelations_avoided / nb_cancelations_due_to_previous_delay_before * 100
        for i in range(8):
            st.text("")
        st.metric(
            label = "", 
            value=f"{round(percent_ended_rentals_removed)}%", 
            delta = "of ended rentals would be lost",
            delta_color = 'inverse'
            )
        st.write()
        for i in range(1):
            st.text("")
        st.metric(
            label = "", 
            value=f"{round(percent_cancelations_avoided)}%",
            delta = "of late checkins leading to cancellations would be avoided",
            delta_color = 'normal'
        )

def apply_threshold(dataframe, threshold):
    condition1 = dataframe['time_delta_with_previous_rental_in_minutes'] >= threshold
    condition2 = dataframe['time_delta_with_previous_rental_in_minutes'].isna()
    mask = condition1 | condition2
    return dataframe[mask]

### Page config
st.set_page_config(
    page_title="GetAround Late Checkouts Analysis",
    page_icon="🚘⏱",
    layout="wide"
)

DATA_PATH = "input_data/get_around_delay_analysis.xlsx"

### Styling
st.write(
    """
    <style>
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

### Title and description
st.title('GetAround Late Checkouts Analysis')
st.markdown("At times, users who rented a car on GetAround are late for their checkout, " \
    "and it can hinder the next rental of the same vehicle, impacting the quality of service and customer satisfaction. 🚘⏱  \n" \
    "A solution for mitigating this problem would be to implement a minimum delay between two rentals " \
    "(by not displaying a car in the search results if the requested checkin or checkout times are too close from an already booked rental).  \n" \
    "As this minimum delay would however impact GetAround and owners' revenues, " \
    "the goal of this analysis is to show some insights that help choosing :  \n" \
    "- The **threshold** (the minimum delay between two rentals) \n" \
    "- The **scope** of application of this threshold (all cars or only 'Connect'\* cars)")
st.caption("_\* 'Connect cars' : the driver doesn’t meet the owner and opens the car with his smartphone_")

@st.cache
def load_data(nrows):
    data = pd.read_excel(DATA_PATH, nrows=nrows)
    return data

raw_data = load_data(None)

### Processing data
data = raw_data.copy()
# modify 'state' column to add information on whether checkout is on time or late :
data['state'] = data.apply(get_checkout_state, axis = 1)
# liken early checkouts (negative delay) to 'on-time' (zero delay) in 'delay_at_checkout_in_minutes' column :
data['delay_at_checkout_in_minutes'] = data['delay_at_checkout_in_minutes'].apply(lambda x : 0 if x < 0 else x)
# add 'previous_rental_checkout_delay_in_minutes' column :
data['previous_rental_checkout_delay_in_minutes'] = data.apply(get_previous_rental_delay, args = [data], axis = 1)
# add 'checkin_delay_in_minutes' column :
data['checkin_delay_in_minutes'] = data['previous_rental_checkout_delay_in_minutes'] - data['time_delta_with_previous_rental_in_minutes']
data['checkin_delay_in_minutes'] = data['checkin_delay_in_minutes'].apply(lambda x : 0 if x < 0 else x)
# add 'impact_of_previous_rental_delay' column :
data['impact_of_previous_rental_delay'] = data.apply(get_impact_of_previous_rental_delay, axis = 1)
# order by rental id :
data = data.sort_values('rental_id')

### Filtered datasets
late_checkout_data = data[data['state'] == 'Late checkout']
connect_data = data[data['checkin_type'] == 'connect']
connect_late_checkout_data = late_checkout_data[late_checkout_data['checkin_type'] == 'connect']
data_with_previous_rental_delay = data[data['previous_rental_checkout_delay_in_minutes'] > 0]
connect_data_with_previous_rental_delay = connect_data[connect_data['previous_rental_checkout_delay_in_minutes'] > 0]
late_checkin_data = data[data['checkin_delay_in_minutes'] > 0]
connect_late_checkin_data = connect_data[connect_data['checkin_delay_in_minutes'] > 0]

### Show data
show_data = st.radio('Show data ?', ['hide', 'show raw data', 'show processed data'])
if show_data == 'show raw data': 
        st.write(raw_data)
        st.write(f"{len(raw_data)} rows")
if show_data == 'show processed data': 
        st.markdown(
            "_Processings made :_\n" \
            "- _modify 'state' column to add information on whether checkout is on time or late,_\n" \
            "- _liken early checkouts (negative delay) to 'on-time' (zero delay)_\n " \
            "- _add 'previous_rental_checkout_delay_in_minutes' column_\n" \
            "- _sort by rental id_"
            )
        st.write(data)
        st.write(f"{len(data)} rows")
st.markdown("_Note: all analyses below are made on processed data_")

### Main metrics
st.header('Main metrics of dataset')
main_metrics_cols = st.columns([1,1,1,1,2])
with main_metrics_cols[0]:
    st.metric(label = "Number of rentals :", value= data['rental_id'].nunique())
with main_metrics_cols[1]:
    st.metric(label = "Number of cars :", value= data['car_id'].nunique())
with main_metrics_cols[2]:
    st.metric(label = "Share of checkout delay outliers :", value= f"{round(detect_outliers(data, 'delay_at_checkout_in_minutes')['percent_dropped'])}%")
with main_metrics_cols[3]:
    st.metric(label = "Share of 'Connect' rentals :", value= f"{round(len(connect_data)/len(data)*100)}%")
with main_metrics_cols[4]:
    st.metric(label = "Share of rentals for which we know the previous one :", value= f"{round(len(data[~data['previous_ended_rental_id'].isna()])/len(data)*100)}%")

### Checkout delays
st.header('Checkouts overview')
checkout_upper_whisker = detect_outliers(late_checkout_data, 'delay_at_checkout_in_minutes')['upper_whisker']
connect_checkout_upper_whisker = detect_outliers(connect_late_checkout_data, 'delay_at_checkout_in_minutes')['upper_whisker']
checkout_delays_overview_scope = st.radio('Scope :', ['All', "'Connect'"], key = 1)
if checkout_delays_overview_scope == 'All':
    show_checkout_delays_overview(data, late_checkout_data, checkout_upper_whisker)
elif checkout_delays_overview_scope == "'Connect'":
    show_checkout_delays_overview(connect_data, connect_late_checkout_data, connect_checkout_upper_whisker)

### Impacts on checkins
st.header('Impacts of delays on next checkin (when existing)')
checkin_upper_whisker = detect_outliers(late_checkin_data, 'checkin_delay_in_minutes')['upper_whisker']
connect_checkin_upper_whisker = detect_outliers(connect_late_checkin_data, 'checkin_delay_in_minutes')['upper_whisker']
impacts_on_checkins_overview_scope = st.radio('Scope :', ['All', "'Connect'"], key = 2)
if impacts_on_checkins_overview_scope == 'All':
    show_impacts_on_checkins_overview(data_with_previous_rental_delay, late_checkin_data, checkin_upper_whisker)
elif impacts_on_checkins_overview_scope == "'Connect'":
    show_impacts_on_checkins_overview(connect_data_with_previous_rental_delay, connect_late_checkin_data, connect_checkin_upper_whisker)

### Simulation form
st.header('Simulation')
with st.form(key='simulation_form'):
    simulation_threshold = st.number_input(label='Threshold :', min_value = 15, step = 15)
    simulation_scope = st.radio('Scope :', ['All', "'Connect'"], key = 3)
    submit = st.form_submit_button(label='Submit')

### Simulation results
# if submit:
#     if simulation_scope == 'All':
#         simulation_data = apply_threshold(data_with_previous_rental_delay, simulation_threshold)
#         whole_simulation_data = apply_threshold(data, simulation_threshold)
#     elif simulation_scope == "'Connect'":
#         simulation_data = apply_threshold(connect_data_with_previous_rental_delay, simulation_threshold)
#         whole_simulation_data = apply_threshold(connect_data, simulation_threshold)
#     show_simulation_results(data_with_previous_rental_delay, simulation_data, data, whole_simulation_data, simulation_scope)
