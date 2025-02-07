#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated Streamlit App for Real-time Balance Check
Integrated with Additional Indicators from balance_group_check.py
"""

# %%
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import requests
import time
import json
import itertools
from scipy import stats
from scipy.stats import ttest_ind
from itertools import combinations
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="CMA Balance Check", 
    page_icon="📊", 
    layout="wide"
)

# Function to fetch and process data from URL
@st.cache_data(ttl=300)  # clear cache every 5 minutes
def fetch_and_process_data(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from {url}. HTTP Status Code: {response.status_code}")

    raw_data = [json.loads(line) for line in response.text.splitlines() if line.strip()]

    clean_tracker = pd.json_normalize(raw_data, sep='.')

    column_mapping = {
        'timestamp': 'timestamp',
        'uuid': 'uuid',
        'event': 'event',
        'data.group': 'group',
        'data.url': 'url',
        'data.sessionCount': 'session_count',
        'data.referrer': 'referrer',
        'data.popupId': 'popup_id',
        'data.amount': 'donation_amount',
        'data.browserInfo.userAgent': 'user_agent',
        'data.browserInfo.language': 'language',
        'data.browserInfo.platform': 'platform',
        'data.browserInfo.screenWidth': 'screen_width',
        'data.browserInfo.screenHeight': 'screen_height',
        'data.browserInfo.windowWidth': 'window_width',
        'data.browserInfo.windowHeight': 'window_height',
        'data.browserInfo.colorDepth': 'color_depth',
        'data.browserInfo.pixelRatio': 'pixel_ratio',
        'data.browserInfo.timezone': 'timezone',
        'data.browserInfo.cookiesEnabled': 'cookies_enabled',
        'data.browserInfo.vendor': 'vendor',
        'data.browserInfo.doNotTrack': 'do_not_track'
    }

    existing_columns = [col for col in column_mapping.keys() if col in clean_tracker.columns]
    clean_tracker = clean_tracker[existing_columns].rename(columns=column_mapping)
    clean_tracker['timestamp'] = pd.to_datetime(clean_tracker['timestamp'], errors='coerce', utc=True)

    # delete testing records: those uuid carrying any url ending with ?track=1
    mask = clean_tracker['url'].str.contains('?track=1', regex=False, na=False)
    track_uuids = clean_tracker.loc[mask, 'uuid'].unique()
    clean_tracker = clean_tracker[~clean_tracker['uuid'].isin(track_uuids)]

    # filter out those non-unique pop-up viewing due to bots and web crawlers
    unique_popup_ids_per_uuid = clean_tracker.groupby('uuid')['popup_id'].nunique()
    non_unique_uuids = unique_popup_ids_per_uuid[unique_popup_ids_per_uuid > 1]
    clean_tracker = clean_tracker[~clean_tracker['uuid'].isin(non_unique_uuids.index)]

    return clean_tracker

# Function to process the tracker data
@st.cache_data
def process_clean_tracker(clean_tracker):
    clean_tracker['standard_group'] = clean_tracker['event'].str.extract(r'(group_v\d+)').ffill()
    clean_tracker['standard_group'].fillna('group_v1', inplace=True)
    clean_tracker['random_group'] = clean_tracker.groupby(['uuid', 'standard_group'])['group'].transform(lambda g: g.ffill().bfill())
    clean_tracker['timestamp'] = pd.to_datetime(clean_tracker['timestamp']).dt.tz_convert('UTC')
    clean_tracker = clean_tracker.sort_values(['uuid', 'timestamp'])
    return clean_tracker

# Generate a sequence number for each session start (replacing session_count)
def generate_session_seq(group):
    session_starts = group[group['event'] == 'session_start'].copy()
    session_starts = session_starts.sort_values('timestamp')
    session_starts['session_seq'] = range(1, len(session_starts) + 1)
    session_starts['session_seq'] = session_starts['session_seq'].astype(int)
    return group.merge(session_starts[['timestamp', 'session_seq']], on='timestamp', how='left')

# Assign session IDs
def assign_sid(group):
    session_starts = group[group['event'] == 'session_start']
    if session_starts.empty:
        return group
    
    session_starts = session_starts.sort_values('timestamp')
    times = session_starts['timestamp'].tolist()
    sids = session_starts['sid'].tolist()
    
    intervals = []
    for i in range(len(times)):
        start = times[i] - pd.Timedelta(seconds=30)
        start = start.tz_convert('UTC')
        
        if i < len(times)-1:
            end = times[i+1] - pd.Timedelta(seconds=30)
            end = end.tz_convert('UTC')
        else:
            end = pd.Timestamp.max.tz_localize('UTC') 
            
        intervals.append((start, end, sids[i]))
    
    group_sorted = group.sort_values('timestamp')
    sids_assigned = []
    for _, row in group_sorted.iterrows():
        t = row['timestamp'].tz_convert('UTC') 
        assigned = False
        for start, end, sid in intervals:
            if start <= t < end:
                sids_assigned.append(sid)
                assigned = True
                break
        if not assigned:
            sids_assigned.append(sids[-1] if sids else np.nan)
    
    group_sorted['sid'] = sids_assigned
    return group_sorted


def process_popup_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Efficiently marks newsletter signup events as popup/website triggered
    """
    # Preprocessing: Sort by session and timestamp
    df_sorted = df.sort_values(['sid', 'timestamp']).copy()
    
    # Add adjacent event info using vectorized operations
    grouped = df_sorted.groupby('sid', group_keys=False)
    df_sorted = df_sorted.assign(
        prev_event=grouped['event'].shift(1),
        next_event=grouped['event'].shift(-1),
        prev_popup_id=grouped['popup_id'].shift(1),
        next_popup_id=grouped['popup_id'].shift(-1)
    )
    
    # Define popup closure conditions
    popup_close_cond = (
        (df_sorted['prev_event'] == 'popup_close') |
        (df_sorted['next_event'] == 'popup_close')
    )
    
    # Mark newsletter signup types
    newsletter_mask = df_sorted['event'] == 'newsletter_signup'
    df_sorted['newsletter_popup'] = newsletter_mask & popup_close_cond
    df_sorted['newsletter_website'] = newsletter_mask & ~popup_close_cond
    
    # Cleanup intermediate columns
    return df_sorted.drop(columns=[
        'prev_event', 'next_event',
        'prev_popup_id', 'next_popup_id'
    ])

def categorize_timezone(tz):
    if tz is None or pd.isna(tz):
        return 'Unknown'
    if 'America/New_York' in tz or 'America/Detroit' in tz or 'America/Indiana' in tz or 'America/Kentucky' in tz or 'America/Port-au-Prince' in tz:
        return 'EST'
    elif 'America/Los_Angeles' in tz or 'America/Santa_Isabel' in tz or 'Pacific/Pitcairn' in tz:
        return 'PST'
    elif 'America/Chicago' in tz or 'America/Winnipeg' in tz or 'America/Mexico_City' in tz or 'America/Guatemala' in tz:
        return 'CST'
    elif 'America/Denver' in tz or 'America/Edmonton' in tz or 'America/Cambridge_Bay' in tz or 'America/Boise' in tz:
        return 'MST'
    else:
        return 'NonUS'

def preprocess_activity_data(clean_tracker):
    # Assign 'sid' for session_start events
    mask = clean_tracker['event'] == 'session_start'
    clean_tracker.loc[mask, 'sid'] = clean_tracker.loc[mask, 'uuid'] + '_s' + clean_tracker.loc[mask, 'session_seq'].astype(int).astype(str)
    clean_tracker = clean_tracker.groupby('uuid', group_keys=False).apply(assign_sid)

    # Reorder columns
    cols = ["timestamp", "uuid", "sid"] + [col for col in clean_tracker.columns if col not in ["timestamp", "uuid", "sid"]]
    clean_tracker = clean_tracker[cols]

    # Fill missing values for specified columns
    cols_to_fill = ['popup_id', 'group', 'random_group', 'user_agent', 
                    'language', 'platform', 'timezone', 'cookies_enabled', 'vendor']
    for col in cols_to_fill:
        clean_tracker[col] = clean_tracker.groupby('sid')[col].transform(lambda x: x.ffill().bfill())

    clean_tracker = process_popup_events(clean_tracker)

    cols_to_fill = ['popup_id', 'group', 'random_group']
    for col in cols_to_fill:
        clean_tracker[col] = clean_tracker.groupby('sid')[col].transform(
            lambda x: x.ffill().bfill().iloc[0] if not x.isnull().all() else np.nan
    )
        
    return clean_tracker


def generate_session_data(clean_tracker):

    def check_url_presence(url_series, keyword):
        return int(any(isinstance(ref, str) and keyword in ref.lower() for ref in url_series))
    
    def check_url_pct(url_series, keyword):
        return (url_series.str.contains(keyword, na=False)).mean()

    agg_dict = {
        'uuid': ('uuid', 'first'),
        'popup_id': ('popup_id', 'first'),
        'session_count_tracker': ('session_count', 'first'),
        'session_count_actual': ('session_seq', 'first'),
        'random_group': ('random_group', 'first'),
        'num_page_views': ('event', lambda x: (x == 'page_view').sum()),
        'num_popup_views': ('event', lambda x: (x == 'popup_view').sum()),
        'num_referral': ('event', lambda x: (x == 'referral').sum()),
        'num_newsletter_signup': ('event', lambda x: (x == 'newsletter_signup').sum()),
        'num_newsletter_signup_popup': ('newsletter_popup', 'sum'),
        'num_newsletter_signup_website': ('newsletter_website', 'sum'),
        'num_donation': ('donation_amount', 'sum'),
        'session_start_time': ('timestamp', 'min'),

        'homepage_pct': ('url', lambda x: check_url_pct(x, r'^https://checkmyads.org/$')),
        'view_about': ('url', lambda x: check_url_presence(x, 'checkmyads.org/about')),
        'view_news': ('url', lambda x: check_url_presence(x, 'checkmyads.org/news')),
        'view_donate': ('url', lambda x: check_url_presence(x, 'checkmyads.org/donate')),
        'view_google_trial': ('url', lambda x: check_url_presence(x, 'checkmyads.org/google')),
        'view_shop': ('url', lambda x: check_url_presence(x, 'checkmyads.org/shop')),

        'referral_google': ('referrer', lambda x: x.str.contains('google', case=False).any()),
        'referral_reddit': ('referrer', lambda x: x.str.contains('reddit', case=False).any()),
        'referral_pcgamer': ('referrer', lambda x: x.str.contains('pcgamer', case=False).any()),
        'referral_globalprivacycontrol': ('referrer', lambda x: x.str.contains('globalprivacycontrol', case=False).any()),
        'referral_duckduckgo': ('referrer', lambda x: x.str.contains('duckduckgo', case=False).any()),
        'referral_bing': ('referrer', lambda x: x.str.contains('bing', case=False).any()),
        'referral_chatgpt': ('referrer', lambda x: x.str.contains('chatgpt', case=False).any()),

        'user_agent': ('user_agent', 'first'),
        'language': ('language', 'first'),
        'platform': ('platform', 'first'),
        'timezone': ('timezone', 'first'),
        'cookies_enabled': ('cookies_enabled', 'first'),
        'vendor': ('vendor', 'first')
    }

    session_tracker = clean_tracker.groupby('sid').agg(**agg_dict).reset_index()

    session_order = clean_tracker.groupby('sid').agg({
        'timestamp': 'min',
        'uuid': 'first'
    }).reset_index()

    session_order = session_order.sort_values(['uuid', 'timestamp'])
    session_order['session_count_recount'] = session_order.groupby('uuid').cumcount() + 1

    session_tracker = session_tracker.merge(
        session_order[['sid', 'session_count_recount']], 
        on='sid'
    )
    session_tracker['timezone_group'] = session_tracker['timezone'].apply(categorize_timezone)
    return session_tracker


# Function to generate user level stats
def generate_user_data(clean_tracker):
    def count_event(event_series, event_name):
        return (event_series == event_name).sum()

    def calculate_homepage_pct(url_series, event_series):
        page_view_count = (event_series == 'page_view').sum()
        if page_view_count > 0:
            return (url_series == 'https://checkmyads.org/').sum() / page_view_count
        return np.nan

    def check_url_presence(url_series, keyword):
        return int(any(isinstance(ref, str) and keyword in ref.lower() for ref in url_series))
    
    def check_url_pct(url_series, keyword):
        return (url_series.str.contains(keyword, na=False)).mean()
    
    agg_dict_session = {
        'num_session_tracker': ('session_count_tracker', 'max'),
        'num_session_actual': ('session_count_actual', 'max'),

        'first_session_start_time': ('session_start_time', 'min'),
        'average_session_start_time': ('session_start_time', 'mean'),
        'last_session_start_time': ('session_start_time', 'max'),
    }

    user_tracker_session = session_tracker.groupby('uuid').agg(**agg_dict_session).reset_index()

    agg_dict_activity = {
        'popup_id': ('popup_id', 'first'),
        'random_group': ('random_group', 'first'),

        'num_page_views': ('event', lambda x: (x == 'page_view').sum()),
        'num_popup_views': ('event', lambda x: (x == 'popup_view').sum()),
        'num_popup_close': ('event', lambda x: (x == 'popup_close').sum()),
        'num_referral': ('event', lambda x: (x == 'referral').sum()),

        'num_newsletter_signup': ('event', lambda x: (x == 'newsletter_signup').sum()),
        'num_newsletter_signup_popup': ('newsletter_popup', 'sum'),
        'num_newsletter_signup_website': ('newsletter_website', 'sum'),
        'num_donation': ('donation_amount', 'sum'),

        'homepage_pct': ('url', lambda x: check_url_pct(x, r'^https://checkmyads.org/$')),
        'view_about': ('url', lambda x: check_url_presence(x, 'checkmyads.org/about')),
        'view_news': ('url', lambda x: check_url_presence(x, 'checkmyads.org/news')),
        'view_donate': ('url', lambda x: check_url_presence(x, 'checkmyads.org/donate')),
        'view_google_trial': ('url', lambda x: check_url_presence(x, 'checkmyads.org/google')),
        'view_shop': ('url', lambda x: check_url_presence(x, 'checkmyads.org/shop')),

        'referral_google': ('referrer', lambda x: x.str.contains('google', case=False).any()),
        'referral_reddit': ('referrer', lambda x: x.str.contains('reddit', case=False).any()),
        'referral_pcgamer': ('referrer', lambda x: x.str.contains('pcgamer', case=False).any()),
        'referral_globalprivacycontrol': ('referrer', lambda x: x.str.contains('globalprivacycontrol', case=False).any()),
        'referral_duckduckgo': ('referrer', lambda x: x.str.contains('duckduckgo', case=False).any()),
        'referral_bing': ('referrer', lambda x: x.str.contains('bing', case=False).any()),
        'referral_chatgpt': ('referrer', lambda x: x.str.contains('chatgpt', case=False).any()),

        'user_agent': ('user_agent', 'first'),  # Since most visitors (except ~2 users) typically experience only one session, during which they are exposed to the popup, we will initially focus on this single session setting for users. We can revisit and adjust this approach later if necessary
        'language': ('language', 'first'),
        'platform': ('platform', 'first'),
        'timezone': ('timezone', 'first'),
        'cookies_enabled': ('cookies_enabled', 'first'),
        'vendor': ('vendor', 'first')
    }
    user_tracker_activity = clean_tracker.groupby('uuid').agg(**agg_dict_activity).reset_index()

    uuid_tracker = pd.merge(user_tracker_activity, user_tracker_session, on=['uuid'], how='outer')
    uuid_tracker['timezone_group'] = uuid_tracker['timezone'].apply(categorize_timezone)
    uuid_tracker['timezone'] = uuid_tracker['timezone'].fillna('Unknown')

    # uuid_tracker['popup_id'] = pd.to_numeric(uuid_tracker['popup_id'], errors='coerce')
    uuid_tracker = uuid_tracker[uuid_tracker['popup_id'] != 4237.0]
    return uuid_tracker

    
# Function to calculate statistics
def calculate_statistics(uuid_tracker):
    agg_dict_stats = {
        'num_uuid': ('uuid', 'nunique'),   
        'num_sessions_mean': ('num_session_actual', 'mean'),
        'num_page_views_mean': ('num_page_views', 'mean'),
        'num_popup_close_mean': ('num_popup_close', 'mean'),
        'num_referral_mean': ('num_referral', 'mean'),

        'num_newsletter_signup_mean': ('num_newsletter_signup', 'mean'),
        'num_newsletter_signup_popup_mean': ('num_newsletter_signup_popup', 'mean'),
        'num_newsletter_signup_website_mean': ('num_newsletter_signup_website', 'mean'),
        'num_donation_mean': ('num_donation', 'mean'),

        'homepage_pct_mean': ('homepage_pct', 'mean'),
        'view_about_mean': ('view_about', 'mean'),
        'view_news_mean': ('view_news', 'mean'),
        'view_donate_mean': ('view_donate', 'mean'),
        'view_google_trial_mean': ('view_google_trial', 'mean'),
        'view_shop_mean': ('view_shop', 'mean'),

        'referral_google_mean': ('referral_google', 'mean'),
        'referral_reddit_mean': ('referral_reddit', 'mean'),
        'referral_pcgamer_mean': ('referral_pcgamer', 'mean'),
        'referral_globalprivacycontrol_mean': ('referral_globalprivacycontrol', 'mean'),
        'referral_duckduckgo_mean': ('referral_duckduckgo', 'mean'),
        'referral_bing_mean': ('referral_bing', 'mean'),
        'referral_chatgpt_mean': ('referral_chatgpt', 'mean'),

        'device_iPhone': ('platform', lambda x: x.str.contains('iPhone', case=False).mean()),
        'device_Mac': ('platform', lambda x: x.str.contains('MacIntel', case=False).mean()),
        'device_Win': ('platform', lambda x: x.str.contains('Win', case=False).mean()),

        'browser_Chrome': ('user_agent', lambda x: ((x.str.contains('Chrome', case=False) & ~x.str.contains('CriOS|Edg', case=False)).mean())),
        'browser_Safari': ('user_agent', lambda x: ((x.str.contains('Safari', case=False) & ~x.str.contains('Chrome', case=False)).mean())),
        'browser_Firefox': ('user_agent', lambda x: x.str.contains('Firefox', case=False).mean()),
        'browser_Edge': ('user_agent', lambda x: ((x.str.contains('Edg', case=False) & ~x.str.contains('Edge;', case=False)).mean())),

        'time_EST': ('timezone_group', lambda x: x.str.contains('EST', case=False).mean()),
        'time_PST': ('timezone_group', lambda x: x.str.contains('PST', case=False).mean()),
        'time_CST': ('timezone_group', lambda x: x.str.contains('CST', case=False).mean()),
        'time_MST': ('timezone_group', lambda x: x.str.contains('MST', case=False).mean()),
        'time_NonUS': ('timezone_group', lambda x: x.str.contains('NonUS', case=False).mean()),

    }

    group_stats = uuid_tracker.groupby('popup_id').agg(**agg_dict_stats).reset_index()
    return group_stats


def datetime_to_numeric(df, datetime_cols):
    for col in datetime_cols:
        # Convert column to datetime first
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Check if already timezone-aware
        if df[col].dt.tz is None:  # tz-naive
            df[col] = df[col].dt.tz_localize('UTC')
        else:  # tz-aware
            df[col] = df[col].dt.tz_convert(None)
        
        # Convert to seconds since epoch
        df[col] = (df[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    return df


def convert_datetime_back(group_stats, datetime_cols):
    for col in datetime_cols:
        # Convert mean back to readable datetime format
        group_stats[(col, 'mean')] = pd.to_datetime(group_stats[(col, 'mean')], unit='s')
        
        # Convert SD back to days (since it was in seconds originally)
        group_stats[(col, 'std')] = group_stats[(col, 'std')] / (60 * 60 * 24)  # Convert seconds to days
    return group_stats


def analyze_user_flow(summary_stats):
    st.subheader("Section 1: User Flow")

    # Map popup_id to new category labels with numeric prefixes
    popup_map = {
        '4217': '1 (Worthiness)',
        '4221': '2 (Numbers)',
        '4223': '3 (Control)'
    }
    summary_stats['popup_label'] = summary_stats['popup_id'].map(popup_map)

    # Filter out unmapped categories and null values
    filtered_data = summary_stats.dropna(subset=['popup_label'])

    # Generate bar chart with correct ordering and legend
    if not filtered_data.empty:
        bar_chart = alt.Chart(filtered_data).mark_bar().encode(
            x=alt.X('popup_label:N', 
                   title='Category',
                   sort=['1 (Worthiness)', '2 (Numbers)', '3 (Control)'], 
                   axis=alt.Axis(labelAngle=0)),
            y=alt.Y('num_uuid:Q', title='Number of UUIDs'),
            color=alt.Color('popup_label:N',
                          legend=alt.Legend(title="Categories"),
                          scale=alt.Scale(
                              domain=['1 (Worthiness)', '2 (Numbers)', '3 (Control)'],
                              range=['blue', 'orange', 'green'])),
            tooltip=[
                alt.Tooltip('num_uuid', title='Number of UUIDs'),
                alt.Tooltip('popup_label', title='Category')
            ]
        ).properties(
            title="Number of Unique Visitors with Each Pop-up",
            height=400
        )

        st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.write("No data available for visualization.")


def words_explanation(uuid_tracker, clean_tracker):
    # get current time
    now = datetime.now(timezone.utc)

    st.markdown(f"""
    **Since Launch**: `{len(uuid_tracker)}` unique user flows
    """)
    
    time_ranges = [
        ("6hrs", timedelta(hours=6)),
        ("24hrs", timedelta(hours=24)),
        ("week", timedelta(weeks=1)),
        ("month", timedelta(days=30))
    ]
    
    for label, delta in time_ranges:
        start_time = now - delta
        count = clean_tracker[clean_tracker['timestamp'] >= start_time]['uuid'].nunique()
        
        st.markdown(f"""
        **Past {label}**: `{count}` active users
        """)

    st.divider()


def analyze_outcome_var(uuid_tracker):
    st.subheader("Section 2: Outcome Variables")

    now = datetime.now(timezone.utc)
    st.markdown(f"""
    **Total email sign up since launch:** `{len(uuid_tracker[uuid_tracker['num_newsletter_signup'] > 0])}`
    """)
    # Worthiness Group: `{len(uuid_tracker[(uuid_tracker['num_newsletter_signup'] > 0) & (uuid_tracker['popup_id'] == 4217)])}` email sign up;
    # Numbers Group: `{len(uuid_tracker[(uuid_tracker['num_newsletter_signup'] > 0) & (uuid_tracker['popup_id'] == 4221)])}` email sign up;
    # Control Group: `{len(uuid_tracker[uuid_tracker['num_newsletter_signup'] > 0 & uuid_tracker['popup_id'] == 4223])}` email sign up;

    st.markdown(f"""
    **Email sign up from pop-ups:** `{len(uuid_tracker[uuid_tracker['num_newsletter_signup_popup'] > 0])}`
    """)

    st.markdown(f"""
    **Email sign up from website:** `{len(uuid_tracker[uuid_tracker['num_newsletter_signup_website'] > 0])}`
    """)

    # Map popup_id to descriptive labels
    popup_map = {
        '4217': '1 (Worthiness)',
        '4221': '2 (Numbers)',
        '4223': '3 (Control)'
    }
    uuid_tracker['popup_label'] = uuid_tracker['popup_id'].map(popup_map)

    # Ensure outcome variables are numeric and handle missing values
    outcome_vars = ['num_newsletter_signup', 
                    'num_newsletter_signup_popup', 
                    'num_newsletter_signup_website', 
                    'num_donation']
    
    for var in outcome_vars:
        uuid_tracker[var] = pd.to_numeric(uuid_tracker[var], errors='coerce')  # Convert to numeric

    # Calculate mean values for each group
    summary_stats = uuid_tracker.groupby('popup_label').agg({
        'num_newsletter_signup': 'mean',
        'num_newsletter_signup_popup': 'mean',
        'num_newsletter_signup_website': 'mean',
        'num_donation': 'mean'
    }).reset_index()

    # Prepare data for Altair bar plot
    plot_data = summary_stats.melt(id_vars=['popup_label'],
                                   value_vars=outcome_vars,
                                   var_name='Outcome Variable',
                                   value_name='Mean Value')

    # Create Altair grouped bar chart
    # chart = alt.Chart(plot_data).mark_bar().encode(
    #     x=alt.X('popup_label:N', title='Popup ID', axis=alt.Axis(labelAngle=0)),
    #     y=alt.Y('Mean Value:Q', title='Mean Value'),
    #     color=alt.Color('Outcome Variable:N', legend=alt.Legend(title="Outcome Variable")),
    #     column=alt.Column('Outcome Variable:N', title=None)
    # ).properties(width=150, height=300)
    # st.altair_chart(chart, use_container_width=True)

    # Perform t-tests between groups for each outcome variable
    results = []
    
    for var in outcome_vars:
        group1 = uuid_tracker[uuid_tracker['popup_id'] == '4217'][var].dropna()  # Filter out NaN
        group2 = uuid_tracker[uuid_tracker['popup_id'] == '4221'][var].dropna()  # Filter out NaN
        group3 = uuid_tracker[uuid_tracker['popup_id'] == '4223'][var].dropna()  # Filter out NaN

        # Perform t-tests between groups
        p_value_1_vs_2 = stats.ttest_ind(group1, group2, equal_var=False).pvalue
        p_value_2_vs_3 = stats.ttest_ind(group2, group3, equal_var=False).pvalue
        p_value_1_vs_3 = stats.ttest_ind(group1, group3, equal_var=False).pvalue

        results.append({
            'Outcome Variable': var,
            '1 (Worthiness): mean': group1.mean(),
            '2 (Numbers): mean': group2.mean(),
            '3 (Control): mean': group3.mean(),
            '1 (Worthiness) vs 2 (Numbers)': p_value_1_vs_2,
            '2 (Numbers) vs 3 (Control)': p_value_2_vs_3,
            '1 (Worthiness) vs 3 (Control)': p_value_1_vs_3,
        })

    # Convert results to DataFrame and display as a table in Streamlit
    results_df = pd.DataFrame(results)
    
    st.write("**Statistical Test Results**")
    st.table(results_df)


def balance_check(df, datetime_cols):
    st.divider()
    st.subheader(f"Section 3: Balance Checks")

    st.markdown(f"**Device Information:**")
    summary_stats_display = summary_stats[['device_iPhone',
       'device_Mac', 'device_Win', 'browser_Chrome', 'browser_Safari',
       'browser_Firefox', 'browser_Edge', 'time_EST', 'time_PST', 'time_CST',
       'time_MST', 'time_NonUS', 'popup_label']]
    summary_stats_display = summary_stats_display.round(3)
    cols = ['popup_label'] + [col for col in summary_stats_display.columns if col != 'popup_label']
    summary_stats_display = summary_stats_display[cols]
    st.dataframe(summary_stats_display)

    # Convert datetime columns to numeric for calculations (seconds precision only)
    df = datetime_to_numeric(df, datetime_cols)

    # Select numeric columns only
    numeric_cols = ['num_page_views', 'num_popup_views', 'num_popup_close', 'num_referral', 
                    'homepage_pct', 'view_about', 'view_news', 'view_donate', 'view_google_trial', 'view_shop', 
                    'referral_google', 'referral_reddit', 'referral_pcgamer', 'referral_globalprivacycontrol',
                    'referral_duckduckgo', 'referral_bing', 'referral_chatgpt',
                    'num_session_tracker', 'num_session_actual', 'first_session_start_time', 'average_session_start_time', 'last_session_start_time']

    print(numeric_cols)
    numeric_df = df[['popup_id'] + numeric_cols]

    # Calculate mean and standard deviation by random group
    group_stats = numeric_df.groupby('popup_id').agg(['mean', 'std'])
    group_stats = convert_datetime_back(group_stats, datetime_cols)  # Apply the conversion back

    # Prepare for pairwise comparisons
    groups = sorted(numeric_df['popup_id'].unique())
    pairwise_results = []

    # Perform pairwise t-tests
    for (group1, group2) in itertools.combinations(groups, 2):
        group1_data = numeric_df[numeric_df['popup_id'] == group1]
        group2_data = numeric_df[numeric_df['popup_id'] == group2]

        for col in numeric_cols:
            stat, p_value = ttest_ind(group1_data[col].dropna(), group2_data[col].dropna(), equal_var=False, nan_policy='omit')
            pairwise_results.append({'Variable': col, 
                                     'Group 1': group1, 
                                     'Group 2': group2, 
                                     'p-value': p_value})

    # Convert results to a dataframe
    pairwise_results_df = pd.DataFrame(pairwise_results)

    # Output tables in an enhanced layout
    for col in numeric_cols:
        # Extract summary statistics for the current characteristic
        summary_stats_col = group_stats[col].reset_index()
        summary_stats_col.columns = ['popup_id', 'Mean', 'SD']

        # Extract pairwise p-values for the current characteristic
        pairwise_p_values_col = pairwise_results_df[pairwise_results_df['Variable'] == col]
        pairwise_p_values_col = pairwise_p_values_col.drop(columns=['Variable'])
        pairwise_p_values_col.reset_index(drop=True, inplace=True)

        # Display both tables side by side
        st.markdown(f"**Balance Check: {col.replace('_', ' ')}**")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f" - *Summary Statistics*")
            if not summary_stats_col.empty:
                summary_stats_col = summary_stats_col.round(3)
                summary_stats_col['popup_id'] = summary_stats_col['popup_id'].astype(str).map(popup_map)

                st.dataframe(summary_stats_col.style.set_table_styles(
                    [
                        {'selector': 'thead', 'props': [('background-color', '#f7f7f7'), ('text-align', 'center')]},
                        {'selector': 'tbody tr:hover', 'props': [('background-color', '#eaf2ff')]}
                    ]
                ))
            else:
                st.write("No data available for this variable.")

        with col2:
            st.markdown(f" - *P-value Comparison*")
            if not pairwise_p_values_col.empty:
                pairwise_p_values_col = pairwise_p_values_col.round(3)
                pairwise_p_values_col['Group 1'] = pairwise_p_values_col['Group 1'].astype(str).map(popup_map)
                pairwise_p_values_col['Group 2'] = pairwise_p_values_col['Group 2'].astype(str).map(popup_map)

                st.dataframe(pairwise_p_values_col.style.set_table_styles(
                    [
                        {'selector': 'thead', 'props': [('background-color', '#f7f7f7'), ('text-align', 'center')]},
                        {'selector': 'tbody tr:hover', 'props': [('background-color', '#eaf2ff')]}
                    ]
                ))
            else:
                st.write("No data available for this variable.")
    return group_stats, pairwise_results_df
    



# URL for fetching data
url = 'https://checkmyads.org/wp-content/themes/checkmyads/tracker-data.txt'
popup_map = {
        '4217': '1 (Worthiness)',
        '4221': '2 (Numbers)',
        '4223': '3 (Control)'
    }

raw_clean_tracker = fetch_and_process_data(url)
raw_clean_tracker = process_clean_tracker(raw_clean_tracker)
raw_clean_tracker = raw_clean_tracker.groupby('uuid', group_keys=False).apply(generate_session_seq)

# Streamlit application setup
st.title("📊 CMA Experiment Monitor")
st.subheader("Please select a randomization version we have tested 🔽")

# Dropdown for selecting test group and select clean_tracker
available_standard_groups = raw_clean_tracker['standard_group'].unique()
available_standard_groups = available_standard_groups[::-1]
selected_standard_group = st.selectbox("Test Group:", options=available_standard_groups)

clean_tracker = raw_clean_tracker[raw_clean_tracker['standard_group'] == selected_standard_group]
# preprocess
clean_tracker = preprocess_activity_data(clean_tracker)
# generate session-level and individual level data
session_tracker = generate_session_data(clean_tracker)
uuid_tracker = generate_user_data(clean_tracker)

# very important step
uuid_tracker = uuid_tracker[uuid_tracker['popup_id'].notna()]


# Section 1: User flows
summary_stats = calculate_statistics(uuid_tracker)
analyze_user_flow(summary_stats)
words_explanation(uuid_tracker, clean_tracker)

# Section 2: Outcome variables
analyze_outcome_var(uuid_tracker)

# Section 3: Balance checks
group_stats, pairwise_results = balance_check(
    uuid_tracker, 
    datetime_cols = ['first_session_start_time', 'average_session_start_time', 'last_session_start_time'])

# %%
