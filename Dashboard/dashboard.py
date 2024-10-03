# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde


# IMPORT DATASET
df = pd.read_csv(r'Dashboard/data.csv')

# DATA MANIPULATION
num_cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
cat_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
df['season'] = df['season'].map({
    1: 'spring',
    2: 'summer',
    3: 'fall',
    4: 'winter'
})
df['yr'] = df['yr'].map({
    0: '2011',
    1: '2012'
})
df['mnth'] = df['mnth'].map({
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'Oktober',
    11: 'November',
    12: 'Desember'
})
df['hr'] = df['hr'].astype('string')
df['holiday'] = df['holiday'].map({
    0: 'not holiday',
    1: 'holiday'
})
df['weekday'] = df['weekday'].map({
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
})
df['workingday'] = df['workingday'].map({
    0: 'not working day',
    1: 'working day'
})
df['weathersit'] = df['weathersit'].astype('string')


# MENU 1
def menu_intro():
    st.header('Introduction to Project')

    tab1, tab2 = st.tabs(['About Dataset', 'Project Goals'])
    with tab1:
        with open(r'Dataset/Readme.txt') as file:
            metadata = file.read()
        st.markdown(metadata)
    with tab2:
        st.subheader('Pertanyaan Bisnis')
        st.markdown('''
            Proyek ini bertujuan untuk melakukan analisis guna menjawab pertanyaan berikut:  
            1. Berapa rata-rata feeling temperature tiap musim yang ramai pengguna rental sepeda?  
            2. Secara umum, bagaimana tren waktu yang ramai pengguna rental sepeda?  
            3. Apakah terdapat hubungan antara frekuensi rental sepeda terhadap keadaan musim/cuaca?
            4. Bagaimana kelompok data jam yang paling banyak pengguna rental sepeda?
        ''')


# MENU 2
def menu_analysis():
    st.header('Exploratory Data Analysis')
    st.divider()

    st.subheader('1. Feeling Temperature Musiman')
    col1, col2 = st.columns(2)
    with col1:
        iSeason = st.multiselect('Season to display', df['season'].unique(), default=df['season'].unique())

        df_season = df.groupby(['season', 'atemp']).agg({'cnt': 'sum'}).reset_index()
        df_season_filt = df_season[df_season['season'].isin(iSeason)]

        chart1 = alt.Chart(df_season_filt).mark_line(point=True).encode(
            x=alt.X('atemp', title='Normalized Feeling Temp.'),
            y=alt.Y('cnt', title='# of Rental Bikes'),
            color=alt.Color('season:N', legend=alt.Legend(title="Season")),
            tooltip=['season', 'atemp', 'cnt']
        ).properties(
            title='Feeling Temp. over Number of Rental Bikes',
            width=800,
            height=400
        ).interactive(bind_x=True)
        st.altair_chart(chart1, use_container_width=True)
    with col2:
        iSeason = st.radio('Select Season', df['season'].unique(), horizontal=True)

        data = df[df['season'] == iSeason]
        grouped_data = data.groupby('atemp').agg({'cnt': 'sum'}).reset_index()

        kde = gaussian_kde(grouped_data['atemp'], weights=grouped_data['cnt'])
        x_range = np.linspace(grouped_data['atemp'].min(), grouped_data['atemp'].max(), 200)
        density = kde(x_range)
        kde_df = pd.DataFrame({'atemp': x_range, 'density': density})
        mean_atemp = grouped_data['atemp'].mean()
        median_atemp = grouped_data['atemp'].median()

        base = alt.Chart(kde_df).mark_area(opacity=0.5, color='steelblue').encode(
            x=alt.X('atemp:Q', title='Normalized Feeling Temp.'),
            y=alt.Y('density:Q', title='Density')
        )
        mean_line = alt.Chart(pd.DataFrame({'value': [mean_atemp]})).mark_rule(color='blue').encode(
            x='value:Q'
        ).properties(title='mean')
        median_line = alt.Chart(pd.DataFrame({'value': [median_atemp]})).mark_rule(color='red', strokeDash=[5, 5]).encode(
            x='value:Q'
        ).properties(title='median')

        chart2 = (base + mean_line + median_line).properties(
            title=f'KDE Plot for {iSeason} season',
            width=600,
            height=400
        )
        st.altair_chart(chart2, use_container_width=True)

    st.text('📝️  Perhatikan bahwa feeling temp. setiap musim cenderung menghampiri distribusi simetris')

    st.divider()

    st.subheader('2. Tren Waktu Pengguna Rental Sepeda')
    col3, col4 = st.columns(2)
    with col3:
        df['dteday'] = pd.to_datetime(df['dteday'])
        data = df.groupby('dteday').agg({'cnt': 'sum'}).reset_index()
        chart3 = alt.Chart(data).mark_line().encode(
            x=alt.X('dteday:T', title=''),
            y=alt.Y('cnt:Q', title='# of Rental Bikes'),
            tooltip=['dteday', 'cnt']
        ).properties(
            title='Number of Rental Bikes Time Series',
            width=800,
            height=500
        ).interactive()
        st.altair_chart(chart3, use_container_width=True)
    with col4:
        data4 = df.groupby('mnth').agg({'cnt': 'mean'}).reset_index()
        chart4 = alt.Chart(data4).mark_line().encode(
            x=alt.X('mnth:O', axis=alt.Axis(labelAngle=-45), title='Monthly'),
            y=alt.Y('cnt:Q', title='# of Rental Bikes'),
            tooltip=['mnth', 'cnt']
        ).properties(
            title='Number of Rental Bikes Trend over Time',
            width=700,
            height=250
        )
        data5 = df.groupby('hr').agg({'cnt': 'mean'}).reset_index()
        chart5 = alt.Chart(data5).mark_line().encode(
            x=alt.X('hr:O', axis=alt.Axis(labelAngle=0), title='Hourly'),
            y=alt.Y('cnt:Q', title='# of Rental Bikes'),
            tooltip=['hr', 'cnt']
        ).properties(
            width=700,
            height=250
        )
        st.altair_chart(chart4, use_container_width=False)
        st.altair_chart(chart5, use_container_width=False)

    st.text('📝️  Pengguna rental sepeda ramai di beberapa waktu tertentu seperti pada 2012-07, dan pada jam 17')

    st.divider()

    st.subheader('3. Hubungan Variabel Terhadap Banyak Pengguna Rental Sepeda')
    tab1, tab2 = st.tabs(['Categorical', 'Numerical'])
    with tab1:
        iRel = st.radio('Select Variable', ['Season', 'Weather'], horizontal=True)
        st.markdown(f'##### Relationship of {iRel} and Number of Rental Bikes')
        if iRel == 'Season':
            x_var = 'season'
        else:
            x_var = 'weathersit'
        col1, col2, _ = st.columns(3)
        with col1:
            sns.boxplot(data=df, x=x_var, y='cnt')
            plt.title('Boxplot of Rental Bikes across Seasons')
            plt.xlabel(iRel)
            plt.ylabel('# of Rental Bikes')
            st.pyplot(plt, use_container_width=True, clear_figure=True)
        with col2:
            sns.violinplot(data=df, x=x_var, y='cnt')
            plt.title('Violinplot of Rental Bikes across Seasons')
            plt.xlabel(iRel)
            plt.ylabel('# of Rental Bikes')
            st.pyplot(plt, use_container_width=True, clear_figure=True)
    with tab2:
        st.markdown(f'##### Correlation Matrix of Numerical Variable')
        col, _ = st.columns(2)
        with col:
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
            st.pyplot(plt, use_container_width=True, clear_figure=True)

    st.text('📝️  Musim spring dan cuaca kategori 3 dan 4 memiliki kecenderungan pengguna rental sepeda yang sedikit dibandingkan dengan lainnya.')

    st.divider()

    with st.popover('Analisis Lanjut: Kelompok Data Waktu dari Banyak Rental Sepeda'):
        df['hr'] = df['hr'].astype('int')
        label = ['Midnight', 'Dawn', 'Morning', 'Afternoon', 'Evening', 'Night']
        df['hr_group'] = pd.cut(df['hr'],
                                bins=[0, 4, 6, 11, 15, 18, 24],
                                labels=label,
                                right=False)
        data = df.groupby('hr_group').agg({'cnt': 'mean'}).reset_index()
        chart6 = alt.Chart(data).mark_bar().encode(
            x=alt.X('hr_group:O', title='', sort=label),
            y=alt.Y('cnt:Q', title='# of Rental Bikes'),
            tooltip=['hr_group', 'cnt']
        ).properties(
            title='Number of Rental Bikes Grouping by Part of Day',
            width=600,
            height=400
        )
        st.altair_chart(chart6)

    st.subheader('Kesimpulan')
    st.markdown('''
    1. Rata-rata feeling temp. yang ramai pengguna rental sepeda pada musim:  
    - spring ≈ 0.3  
    - summer ≈ 0.5  
    - fall ≈ 0.67  
    - winter ≈ 0.41.  
    2. Pengguna rental sepeda semakin banyak pada periode sekitar Mei-Oktober 2012 hingga akhirnya menurun pada November-Desember 2012,  
    dan peak hours pengguna rental sepeda secara rata-rata adalah di pagi jam 7.00 dan sore jam 17.00, yaitu jam pergi dan pulang kerja.  
    3. Keadaan musim dan cuaca berpengaruh terhadap banyaknya pengguna rental sepeda, hal ini dikuatkan oleh uji ANOVA yang menolak $H_0$.  
    4. Secara rata-rata, pengguna rental sepeda paling banyak di kelompok waktu sore (di antara jam 15.00 - 18.00).
    ''')

# CONFIG PAGE SETUP
def set_page_configuration():
    st.set_page_config(
        page_title='Bike Sharing Data Analysis',
        page_icon='🚲',
        layout='wide',
        initial_sidebar_state='expanded')


# NAV MENU
def navbar_menu():
    with st.sidebar:
        selected_navbar_menu = option_menu(
            menu_title='Menu Page',
            options=['Introduction', 'Analysis'],
            icons=['info-square', 'graph-up-arrow'],
            menu_icon='cast',
            default_index=0,
            orientation='vertical')

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("""
                    Developed by:    
                    - Fery Kurniawan
                    """)
        st.caption('Bangkit Academy Course Project')
        st.text('*Close this sidebar for better layout*')

        return selected_navbar_menu


def main():
    selected_navbar_menu = navbar_menu()
    if selected_navbar_menu == 'Introduction':
        menu_intro()
    if selected_navbar_menu == 'Analysis':
        menu_analysis()


# Run Program
if __name__ == '__main__':
    set_page_configuration()
    main()
