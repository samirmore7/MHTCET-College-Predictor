import os
import gzip
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="EduPredict.AI — World Class Admission Engine",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. MULTI-THEME ENGINE (10 DYNAMIC PRESETS)
# ==============================================================================
if 'current_theme' not in st.session_state:
    st.session_state.current_theme = "Light Mode"

THEME_CONFIGS = {
    "Light Mode": {
        "bg_base": "#F8FAFC",
        "card_bg": "rgba(255, 255, 255, 0.85)",
        "card_border": "rgba(0, 0, 0, 0.08)",
        "text_main": "#0F172A",
        "text_muted": "#64748B",
        "primary": "#4F46E5",
        "secondary": "#DB2777",
        "accent": "#0891B2",
        "glow": "rgba(79, 70, 229, 0.15)"
    },
    "Dark Cyberpunk": {
        "bg_base": "#060913",
        "card_bg": "rgba(15, 23, 42, 0.75)",
        "card_border": "rgba(255, 255, 255, 0.08)",
        "text_main": "#F8FAFC",
        "text_muted": "#94A3B8",
        "primary": "#6366F1",
        "secondary": "#EC4899",
        "accent": "#06B6D4",
        "glow": "rgba(99, 102, 241, 0.45)"
    },
    "Luxe Gold": {
        "bg_base": "#0B0A07",
        "card_bg": "rgba(26, 22, 16, 0.75)",
        "card_border": "rgba(234, 179, 8, 0.15)",
        "text_main": "#FEF08A",
        "text_muted": "#CA8A04",
        "primary": "#EAB308",
        "secondary": "#F97316",
        "accent": "#FACC15",
        "glow": "rgba(234, 179, 8, 0.35)"
    },
    "Emerald Luxe": {
        "bg_base": "#022C22",
        "card_bg": "rgba(6, 78, 59, 0.65)",
        "card_border": "rgba(52, 211, 153, 0.2)",
        "text_main": "#ECFDF5",
        "text_muted": "#6EE7B7",
        "primary": "#10B981",
        "secondary": "#059669",
        "accent": "#34D399",
        "glow": "rgba(16, 185, 129, 0.35)"
    },
    "Dracula Night": {
        "bg_base": "#282A36",
        "card_bg": "rgba(68, 71, 90, 0.7)",
        "card_border": "rgba(189, 147, 249, 0.2)",
        "text_main": "#F8F8F2",
        "text_muted": "#6272A4",
        "primary": "#BD93F9",
        "secondary": "#FF79C6",
        "accent": "#8BE9FD",
        "glow": "rgba(189, 147, 249, 0.3)"
    },
    "Neon Vaporwave": {
        "bg_base": "#180226",
        "card_bg": "rgba(48, 10, 72, 0.7)",
        "card_border": "rgba(255, 0, 127, 0.25)",
        "text_main": "#FFE5F1",
        "text_muted": "#D980FA",
        "primary": "#FF007F",
        "secondary": "#00F0FF",
        "accent": "#FFE600",
        "glow": "rgba(255, 0, 127, 0.4)"
    },
    "Nordic Frost": {
        "bg_base": "#2E3440",
        "card_bg": "rgba(59, 66, 82, 0.8)",
        "card_border": "rgba(136, 192, 208, 0.2)",
        "text_main": "#ECEFF4",
        "text_muted": "#D8DEE9",
        "primary": "#88C0D0",
        "secondary": "#81A1C1",
        "accent": "#8FBCBB",
        "glow": "rgba(136, 192, 208, 0.25)"
    },
    "Sunset Amber": {
        "bg_base": "#1F110B",
        "card_bg": "rgba(67, 30, 15, 0.7)",
        "card_border": "rgba(251, 146, 60, 0.2)",
        "text_main": "#FFEDD5",
        "text_muted": "#FDBA74",
        "primary": "#F97316",
        "secondary": "#EF4444",
        "accent": "#FBBF24",
        "glow": "rgba(249, 115, 22, 0.35)"
    },
    "Deep Ocean": {
        "bg_base": "#031B2E",
        "card_bg": "rgba(11, 43, 70, 0.75)",
        "card_border": "rgba(56, 189, 248, 0.2)",
        "text_main": "#E0F2FE",
        "text_muted": "#7DD3FC",
        "primary": "#0284C7",
        "secondary": "#06B6D4",
        "accent": "#38BDF8",
        "glow": "rgba(2, 132, 199, 0.35)"
    },
    "Monochrome Obsidian": {
        "bg_base": "#000000",
        "card_bg": "rgba(20, 20, 20, 0.85)",
        "card_border": "rgba(255, 255, 255, 0.15)",
        "text_main": "#FFFFFF",
        "text_muted": "#A1A1AA",
        "primary": "#E4E4E7",
        "secondary": "#71717A",
        "accent": "#FAFAFA",
        "glow": "rgba(255, 255, 255, 0.2)"
    }
}

# Sidebar Theme Selector Switcher
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("Theme Selector")
    selected_theme_name = st.selectbox(
        "Choose Dashboard Theme", 
        list(THEME_CONFIGS.keys()), 
        index=list(THEME_CONFIGS.keys()).index(st.session_state.current_theme)
    )
    st.session_state.current_theme = selected_theme_name

theme = THEME_CONFIGS[st.session_state.current_theme]

# Dynamic CSS Injection for Ultra-Glassmorphism UI
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {theme['bg_base']};
            color: {theme['text_main']};
            background-image: 
                radial-gradient(circle at 10% 10%, {theme['glow']} 0%, transparent 45%),
                radial-gradient(circle at 90% 90%, {theme['glow']} 0%, transparent 45%);
            background-attachment: fixed;
        }}
        .glass-card {{
            background: {theme['card_bg']};
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid {theme['card_border']};
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
            margin-bottom: 25px;
        }}
        .hero-prediction-box {{
            background: {theme['card_bg']};
            border: 2px solid {theme['primary']};
            box-shadow: 0 0 25px {theme['glow']};
            border-radius: 18px;
            padding: 25px;
            text-align: center;
            margin-bottom: 25px;
        }}
        .prediction-title-text {{
            color: {theme['accent']};
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: 800;
            margin-bottom: 8px;
        }}
        .prediction-value-text {{
            color: {theme['text_main']};
            font-size: 1.8rem;
            font-weight: 800;
        }}
        .metric-pill {{
            background: rgba(0,0,0,0.2);
            border: 1px solid {theme['card_border']};
            border-radius: 12px;
            padding: 12px 18px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .stButton>button {{
            background: linear-gradient(135deg, {theme['primary']}, {theme['secondary']});
            color: #FFFFFF !important;
            border: none;
            border-radius: 12px;
            padding: 16px 28px;
            font-size: 1.05rem;
            font-weight: 700;
            box-shadow: 0 10px 25px {theme['glow']};
            transition: all 0.3s ease;
            width: 100%;
        }}
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 15px 35px {theme['glow']};
        }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. COMPREHENSIVE NATIVE DATASETS
# ==============================================================================

FALLBACK_GENDERS = ["Female", "Male"]

FALLBACK_CATEGORIES = [
    "DT/VJ", "DT/VJ#", "DT/VJ$", "DT/VJ$#", "DT/VJ$/DEF1", "DT/VJ$/DEF2", "DT/VJ$/PH1",
    "DT/VJ/DEF1", "DT/VJ/DEF2", "DT/VJ/PH1", "NT 1 (NT-B)", "NT 1 (NT-B)#", 
    "NT 1 (NT-B)#/PH1", "NT 1 (NT-B)$", "NT 1 (NT-B)$#", "NT 1 (NT-B)/DEF1", 
    "NT 1 (NT-B)/DEF2", "NT 1 (NT-B)/PH1", "NT 2 (NT-C)", "NT 2 (NT-C)#", 
    "NT 2 (NT-C)#/PH1", "NT 2 (NT-C)$", "NT 2 (NT-C)$#", "NT 2 (NT-C)/DEF1", 
    "NT 2 (NT-C)/DEF2", "NT 2 (NT-C)/PH1", "NT 3 (NT-D)", "NT 3 (NT-D)#", 
    "NT 3 (NT-D)#/PH1", "NT 3 (NT-D)$", "NT 3 (NT-D)$#", "NT 3 (NT-D)/DEF1", 
    "NT 3 (NT-D)/DEF2", "NT 3 (NT-D)/PH1", "OBC", "OBC#", "OBC#/DEF1", "OBC#/DEF2", 
    "OBC$", "OBC$#", "OBC$#/DEF1", "OBC$#/DEF2", "OBC$#/PH1", "OBC$/DEF1", "OBC$/DEF2", 
    "OBC$/DEF3", "OBC$/PH1", "OBC/DEF1", "OBC/DEF2", "OBC/PH1", "OPEN", "OPEN@", 
    "Open/DEF1", "Open/DEF2", "Open/DEF3", "Open/PH1", "Open/PH1/DEF1", "Open@/DEF1", 
    "Open@/PH1", "SBC", "SBC#", "SBC#/DEF1", "SBC$", "SBC$#", "SBC$/DEF1", "SBC/DEF1", 
    "SBC/DEF3", "SBC/PH1", "SC", "SC$", "SC$/DEF1", "SC/DEF1", "SC/DEF2", "SC/DEF3", 
    "SC/PH1", "SEBC", "SEBC#", "SEBC$", "SEBC$#", "SEBC$#/DEF1", "SEBC$#/DEF2", 
    "SEBC$#/PH1", "SEBC$/DEF1", "SEBC$/DEF2", "SEBC$/PH1", "SEBC/DEF1", "SEBC/PH1", 
    "ST", "ST$", "ST$/DEF1", "ST/DEF1", "ST/DEF2", "ST/PH1"
]

FALLBACK_SEATS = [
    "DEFOBCS", "DEFOPENS", "DEFRNT1S", "DEFRNT2S", "DEFRNT3S", "DEFROBCS", "DEFRSCS",
    "DEFRSEBCS", "DEFRVJS", "DEFSCS", "DEFSEBCS", "EWS", "GNT1H", "GNT1O", "GNT1S",
    "GNT2H", "GNT2O", "GNT2S", "GNT3H", "GNT3O", "GNT3S", "GOBCH", "GOBCO", "GOBCS",
    "GOPENH", "GOPENO", "GOPENS", "GSCH", "GSCO", "GSCS", "GSEBCH", "GSEBCO", "GSEBCS",
    "GSTH", "GSTO", "GSTS", "GVJH", "GVJO", "GVJS", "LNT1H", "LNT1O", "LNT1S",
    "LNT2H", "LNT2O", "LNT2S", "LNT3H", "LNT3O", "LNT3S", "LOBCH", "LOBCO", "LOBCS",
    "LOPENH", "LOPENO", "LOPENS", "LSCH", "LSCO", "LSCS", "LSEBCH", "LSEBCO", "LSEBCS",
    "LSTH", "LSTO", "LSTS", "LVJH", "LVJO", "LVJS", "MI", "ORPHAN", "PWDOBCH",
    "PWDOBCS", "PWDOPENH", "PWDOPENS", "PWDRNT2S", "PWDRNT3S", "PWDROBCH", "PWDROBCS",
    "PWDRSCS", "PWDRSEBCS", "PWDRSTS", "PWDRVJS", "PWDSCS", "PWDSEBCS", "TFWS"
]

FALLBACK_COURSES = [
    "5G", "Aeronautical Engineering", "Agricultural Engineering", "Artificial Intelligence",
    "Artificial Intelligence (AI) and Data Science", "Artificial Intelligence and Data Science",
    "Artificial Intelligence and Machine Learning", "Automation and Robotics", "Automobile Engineering",
    "Bio Medical Engineering", "Bio Technology", "Chemical Engineering", "Civil Engineering",
    "Civil Engineering and Planning", "Civil and Environmental Engineering", "Civil and infrastructure Engineering",
    "Computer Engineering", "Computer Engineering (Software Engineering)", "Computer Science",
    "Computer Science and Business Systems", "Computer Science and Design", "Computer Science and Engineering",
    "Computer Science and Engineering (Artificial Intelligence and Data Science)",
    "Computer Science and Engineering (Artificial Intelligence)", "Computer Science and Engineering (Cyber Security)",
    "Computer Science and Engineering (Internet of Things and Cyber Security Including Block Chain Technology)",
    "Computer Science and Engineering (IoT)", "Computer Science and Engineering(Artificial Intelligence and Machine Learning)",
    "Computer Science and Engineering(Cyber Security)", "Computer Science and Engineering(Data Science)",
    "Computer Science and Information Technology", "Computer Science and Technology", "Computer Technology",
    "Cyber Security", "Data Engineering", "Data Science", "Dyestuff Technology",
    "Electrical Engg[Electronics and Power]", "Electrical Engineering", "Electrical and Computer Engineering",
    "Electrical and Electronics Engineering", "Electronics Engineering", "Electronics Engineering ( VLSI Design and Technology)",
    "Electronics and Biomedical Engineering", "Electronics and Communication (Advanced Communication Technology)",
    "Electronics and Communication Engineering", "Electronics and Communication(Advanced Communication Technology)",
    "Electronics and Computer Engineering", "Electronics and Computer Science", "Electronics and Telecommunication Engg",
    "Fashion Technology", "Fibres and Textile Processing Technology", "Fire Engineering", "Food Engineering and Technology",
    "Food Technology", "Food Technology And Management", "Industrial IoT", "Information Technology",
    "Instrumentation Engineering", "Instrumentation and Control Engineering", "Internet of Things (IoT)",
    "Man Made Textile Technology", "Manufacturing Science and Engineering", "Mechanical & Automation Engineering",
    "Mechanical Engineering", "Mechanical Engineering Automobile", "Mechanical Engineering[Sandwich]",
    "Mechanical and Mechatronics Engineering (Additive Manufacturing)", "Mechatronics Engineering",
    "Metallurgy and Material Technology", "Mining Engineering", "Oil Technology", "Oil and Paints Technology",
    "Oil,Oleochemicals and Surfactants Technology", "Paints Technology", "Paper and Pulp Technology",
    "Petro Chemical Engineering", "Pharmaceutical and Fine Chemical Technology", "Pharmaceuticals Chemistry and Technology",
    "Plastic Technology", "Plastic and Polymer Engineering", "Polymer Engineering and Technology",
    "Printing and Packing Technology", "Production Engineering", "Production Engineering[Sandwich]",
    "Robotics and Artificial Intelligence", "Robotics and Automation", "Safety and Fire Engineering",
    "Structural Engineering", "Surface Coating Technology", "Technical Textiles", "Textile Chemistry",
    "Textile Engineering / Technology", "Textile Technology", "VLSI"
]

FALLBACK_INSTITUTES = [
    "A. V. College of Engineering and Technology, Dongargaon, Nagpur",
    "A. P. Shah Institute of Technology, Thane",
    "APPASAHEB ALIAS SA.RE.PATIL INSTITUTE OF TECHNOLOGY, Dattanagar  Tal-Shirol, Dist Kolhapur",
    "Abhinav Education Society's College of Engineering and Technology (Degree), Wadwadi",
    "Aditya Education Trust's Mitthulalji Sarada Polytechnic, Nalwandi Road, Beed",
    "Aditya Engineering College , Beed",
    "Adsul's Technical Campus, Chas Dist. Ahmednagar",
    "Agnel Charities' FR. C. Rodrigues Institute of Technology, Vashi, Navi Mumbai",
    "Ahmednagar Jilha Maratha Vidya Prasarak Samajache, Shri. Chhatrapati Shivaji Maharaj College of Engineering, Nepti",
    "Ajeenkya DY Patil School of Engineering, Lohegaon, Pune",
    "Al-Ameen Educational and Medical Foundation, College of Engineering, Koregaon, Bhima",
    "Alard Charitable Trust's Alard College of Engineering and Management, Pune",
    "Aldel Education Trust's St. John College of Engineering & Management, Vevoor, Palghar",
    "All India Shri Shivaji Memorial Society's College of Engineering, Pune",
    "All India Shri Shivaji Memorial Society's Institute of Information Technology,Pune",
    "Amar Seva Mandal's Shree Govindrao Vanjari College of Engineering & Technology, Nagpur",
    "Amruta Vaishnavi Education & Welfare Trust's Shatabdi Institute of Engineering & Research, Agaskhind Tal. Sinnar",
    "Amrutvahini Sheti & Shikshan Vikas Sanstha's Amrutvahini College of Engineering, Sangamner",
    "Anantrao Pawar College of Engineering & Research, Pune",
    "Anjuman College of Engineering & Technology, Nagpur",
    "Anjuman-I-Islam's Kalsekar Technical Campus, Panvel",
    "Anjuman-I-Islam's M.H. Saboo Siddik College of Engineering, Byculla, Mumbai",
    "Ankush Shikshan Sanstha's G.H.Raisoni College of Engineering, Nagpur",
    "Annasaheb Dange College of Engineering and Technology, Ashta, Sangli",
    "Atharva College of Engineering,Malad(West),Mumbai",
    "Aurangabad College of Engineering, Naygaon Savangi, Aurangabad",
    "B.R. Harne College of Engineering & Technology, Karav, Tal-Ambernath.",
    "Bajaj Institute of Technology, Wardha",
    "Bansilal Ramnath Agarawal Charitable Trust's Vishwakarma Institute of Technology, Bibwewadi, Pune",
    "Bapurao Deshmukh College of Engineering, Sevagram",
    "Bhagwant Institute of Technology, Barshi",
    "Bharat College of Engineering, Kanhor, Badlapur(W)",
    "Bharati Vidyapeeth College of Engineering, Navi Mumbai",
    "Bharati Vidyapeeth's College of Engineering for Women, Katraj, Dhankawadi, Pune",
    "Bharati Vidyapeeth's College of Engineering, Kolhapur",
    "Bharati Vidyapeeth's College of Engineering,Lavale, Pune",
    "Bhartiya Vidya Bhavan's Sardar Patel Institute of Technology , Andheri, Mumbai",
    "Brahma Valley College of Engineering & Research, Trimbakeshwar, Nashik",
    "COEP Technological University",
    "CSMSS Chh. Shahu College of Engineering, Aurangabad",
    "Chhartrapati Shivaji Maharaj Institute of Technology, Shedung, Panvel",
    "College of Engineering and Technology ,North Maharashtra Knowledge City, Jalgaon",
    "Cummins College of Engineering For Women, Sukhali (Gupchup), Tal. Hingna Hingna Nagpur",
    "D.Y. Patil College of Engineering and Technology, Kolhapur",
    "D.Y.Patil Education Society's,D.Y.Patil Technical Campus, Faculty of Engineering & Faculty of Management,Talsande,Kolhapur.",
    "Dattajirao Kadam Technical Education Society's Textile & Engineering Institute, Ichalkaranji.",
    "Dattakala Group Of Institutions, Swami - Chincholi Tal. Daund Dist. Pune",
    "Deogiri Institute of Engineering and Management Studies, Aurangabad",
    "Department of Technology, Shivaji University, Kolhapur",
    "Devi Mahalaxmi College of Engineering and Technology",
    "Dhole Patil Education Society, Dhole Patil College of Engineering, Wagholi, Tal. Haveli",
    "Dilkap Research Institute Of Engineering and Management Studies, At.Mamdapur, Post- Neral, Tal- Karjat, Mumbai",
    "Dnyanshree Institute Engineering and Technology, Satara",
    "Don Bosco Institute of Technology, Mumbai",
    "Dr. A. D. Shinde College Of Engineering, Tal.Gadhinglaj, Kolhapur",
    "Dr. Ashok Gujar Technical Institute's Dr. Daulatrao Aher College of Engineering, Karad",
    "Dr. Babasaheb Ambedkar Technological University, Lonere",
    "Dr. D Y Patil Pratishthan's College of Engineering, Kolhapur",
    "Dr. D. Y. Patil Pratishthan's D.Y.Patil College of Engineering Akurdi, Pune",
    "Dr. D. Y. Patil Unitech Society's Dr. D. Y. Patil Institute of Technology, Pimpri, Pune",
    "Dr. D.Y. Patil Technical Campus, Varale, Talegaon, Pune",
    "Dr. J. J. Magdum Charitable Trust's Dr. J.J. Magdum College of Engineering, Jaysingpur",
    "Dr. V.K. Patil College of Engineering & Technology",
    "Dr. Vithalrao Vikhe Patil College of Engineering, Ahmednagar",
    "Dr.D.Y.Patil College Of Engineering & Innovation,Talegaon",
    "Dr.Rajendra Gode Institute of Technology & Research, Amravati",
    "Dwarka Bahu Uddeshiya Gramin Vikas Foundation, Rajarshi Shahu College of Engineering, Buldhana",
    "Everest Education Society, Group of Institutions (Integrated Campus), Ohar",
    "Excelsior Education Society's K.C. College of Engineering and Management Studies and Research, Kopri, Thane (E)",
    "Fabtech Technical Campus College of Engineering and Research, Sangola",
    "Flora Institute of Technology, Khopi, Near Khed Shivapur Toll Plaza, Pune",
    "Fr. Conceicao Rodrigues College of Engineering, Bandra,Mumbai",
    "G H Raisoni College of Engineering and Management, Jalgaon",
    "G H Raisoni College of Engineering & Management, Nagpur",
    "G. S. Mandal's Maharashtra Institute of Technology, Aurangabad",
    "G.H.Raisoni College of Engineering & Management, Wagholi, Pune",
    "G.M.Vedak Institute of Technology, Tala, Raigad.",
    "GRAMIN TECHNICAL AND MANAGEMENT CAMPUS NANDED.",
    "Genba Sopanrao Moze College of Engineering, Baner-Balewadi, Pune",
    "Genba Sopanrao Moze Trust Parvatibai Genba Moze College of Engineering,Wagholi, Pune",
    "Gharda Foundation's Gharda Institute of Technology,Khed, Ratnagiri",
    "Godavari Foundation's Godavari College Of Engineering, Jalgaon",
    "Gokhale Education Society's, R.H. Sapat College of Engineering, Management Studies and Research, Nashik",
    "Gondia Education Society's Manoharbhai Patel Institute Of Engineering & Technology, Shahapur, Bhandara",
    "Government College of Engineering & Research, Avasari Khurd",
    "Government College of Engineering, Chhatrapati Sambhajinagar",
    "Government College of Engineering, Amravati",
    "Government College of Engineering, Chandrapur",
    "Government College of Engineering, Jalgaon",
    "Government College of Engineering, Karad",
    "Government College of Engineering, Kolhapur",
    "Government College of Engineering, Nagpur",
    "Government College of Engineering,Yavatmal",
    "Guru Gobind Singh College of Engineering & Research Centre, Nashik.",
    "Guru Nanak Institute of Engineering & Technology,Kalmeshwar, Nagpur",
    "Gurunanak Educational Society's Gurunanak Institute of Technology, Nagpur",
    "Haji Jamaluddin Thim Trust's Theem College of Engineering, At. Villege Betegaon, Boisar",
    "Hi-Tech Institute of Technology, Aurangabad",
    "Hindi Seva Mandal's Shri Sant Gadgebaba College of Engineering & Technology, Bhusawal",
    "Hon. Shri. Babanrao Pachpute Vichardhara Trust, Group of Institutions (Integrated Campus)-Parikrama, Kashti Shrigondha",
    "Hope Foundation and research center's Finolex Academy of Management and Technology, Ratnagiri",
    "ISBM College Of Engineering Pune",
    "Ideal Institute of Technology, Wada, Dist.Thane",
    "Indala College Of Engineering, Bapsai Tal.Kalyan",
    "Indira College of Engineering & Management, Pune",
    "Institute of Chemical Technology, Matunga, Mumbai",
    "Institute of Chemical Technology, Mumbai Marathwada off campus, Jalna",
    "International Centre Of Excellence In Engineering and Management (ICEEM)",
    "International Institute of Information Technology (I2IT), Pune",
    "JMSS Shri Shankarprasad Agnihotri College of Engineering, Wardha",
    "JSPM Narhe Technical Campus, Pune.",
    "JSPM'S Jaywantrao Sawant College of Engineering,Pune",
    "Jagadamba Education Soc. Nashik's S.N.D. College of Engineering & Reserch, Babulgaon",
    "Jagadambha Bahuuddeshiya Gramin Vikas Sanstha's Jagdambha College of Engineering and Technology, Yavatmal",
    "Jaidev Education Society, J D College of Engineering and Management, Nagpur",
    "Jaihind College Of Engineering,Kuran",
    "Janata Shikshan Prasarak Mandal's Babasaheb Naik College Of Engineering, Pusad",
    "Jawahar Education Society's Annasaheb Chudaman Patil College of Engineering,Kharghar, Navi Mumbai",
    "Jawahar Education Society's Institute of Technology, Management & Research, Nashik.",
    "Jawaharlal Darda Institute of Engineering and Technology, Yavatmal",
    "Jayawant Shikshan Prasarak Mandal, Bhivarabai Sawant Institute of Technology & Research, Wagholi",
    "Jaywant College of Engineering & Polytechnic , Kille Macchindragad Tal. Walva District- Sangali",
    "Jaywant Shikshan Prasarak Mandal's,Rajarshi Shahu College of Engineering, Tathawade, Pune",
    "Jijau Institute of Engineering Technology and Management, Khandgaon (Bendri), Taluka Naigaon, District Nanded",
    "K J Somaiya Institute of Technology",
    "K. E. Society's Rajarambapu Institute of Technology, Walwa, Sangli",
    "K. J.'s Educational Institut Trinity College of Engineering and Research, Pisoli, Haveli",
    "K. K. Wagh Institute of Engineering Education and Research, Nashik",
    "K.D.K. College of Engineering, Nagpur",
    "K.D.M. Education Society, Vidharbha Institute of Technology,Umred Road ,Nagpur",
    "K.J.'s Educational Institute's K.J.College of Engineering & Management Research, Pisoli",
    "K.V.N. Naik S. P. Sansth's Loknete Gopinathji Munde Institute of Engineering Education & Research, Nashik.",
    "KJEI's Trinity Academy of Engineering, Yewalewadi, Pune",
    "KSGBS's Bharat- Ratna Indira Gandhi College of Engineering, Kegaon, Solapur",
    "Kai Amdar Bramhadevdada Mane Shikshan & Samajik Prathistan's Bramhadevdada Mane Institute of Technology, Solapur",
    "Kalyani Charitable Trust, Late Gambhirrao Natuba Sapkal College of Engineering, Anjaneri, Trimbakeshwar Road, Nashik",
    "Karanjekar College of Engineering & Management, Sakoli",
    "Karmayogi Institute of Technology",
    "Kavi Kulguru Institute of Technology & Science, Ramtek",
    "Kedareshwar Gramin Vikas Pratishthan, Samajbhushan Eknathrao Dhakane College, of Engineering, Shevgaon",
    "Keystone School of Engineering, Pune",
    "Khandesh College Education Society's College Of Engineering And Management, Jalgaon",
    "Kolhapur Institute of Technology's College of Engineering(Autonomous), Kolhapur",
    "Konkan Gyanpeeth College of Engineering, Karjat",
    "Koti Vidya Charitable Trust's Smt. Alamuri Ratnamala Institute of Engineering and Technology, Sapgaon, Tal. Shahapur",
    "Krushi Jivan Vikas Pratishthan, Ballarpur Institute of Technology, Mouza Bamni",
    "Late Narayandas Bhawandas Chhabada Institute of Engineering & Technology, Satara",
    "Laxminarayan Innovation Technological University, Nagpur",
    "Leela Education Society, G.V. Acharya Institute of Engineering and Technology, Shelu, Karjat",
    "Lokmanya Tilak College of Engineering, Kopar Khairane, Navi Mumbai",
    "Lokmanya Tilak Jankalyan Shikshan Sanstha, Priyadarshani College of Engineering, Nagpur",
    "Lokmanya Tilak Jankalyan Shikshan Sastha, Priyadarshini J. L. College Of Engineering, Nagpur",
    "Loknete Hanumantrao Charitable Trust's Adarsh Institute of Technology and Research Centre, Vita,Sangli",
    "Loknete Shamrao Peje Government College of Engineering, Ratnagiri",
    "M.D. Yergude Memorial Shikshan Prasarak Mandal's Shri Sai College of Engineering & Technology, Bhadrawati",
    "M.G.M.'s College of Engineering and Technology, Kamothe, Navi Mumbai",
    "M.S. Bidve Engineering College, Latur",
    "MAEER's MIT College of Railway Engineering and Research, Jamgaon, Barshi",
    "MET Bhujbal Knowledge City MET League's Engineering College, Adgaon, Nashik.",
    "MET's Institute of Technology Polytechnic, Bhujbal Knowledge City, Adgaon Nashik",
    "MIT Academy of Engineering,Alandi, Pune",
    "MKD Institute of Technology, Nadurbar",
    "MKSSS's Cummins College of Engineering for Women, Karvenagar,Pune",
    "Mahatma Basaweshwar Education Society's College of Engineering, Ambejogai",
    "Mahatma Education Society's Pillai College of Engineering, New Panvel",
    "Mahatma Education Society's Pillai HOC College of Engineering & Technology, Tal. Khalapur. Dist. Raigad",
    "Mahatma Gandhi Missions College of Engineering, Hingoli Rd, Nanded.",
    "Mahavir Education Trust's Shah & Anchor Kutchhi Engineering College, Mumbai",
    "Maitraya Education Society, Nagarjuna Institute of Engineering Technology & Management, Nagpur",
    "Manav School of Engineering & Technology, Gut No. 1035 Nagpur Surat Highway, NH No. 6 Tal.Vyala, Balapur, Akola, 444302",
    "Mangaldeep College of Engineering",
    "Manjara Charitable Trust's Rajiv Gandhi Institute of Technology, Mumbai",
    "Maratha Vidya Prasarak Samaj's Karmaveer Adv. Baburao Ganpatrao Thakare College Of Engineering, Nashik",
    "Marathwada Mitra Mandal's College of Engineering, Karvenagar, Pune",
    "Marathwada Mitra Mandal's Institute of Technology, Lohgaon, Pune",
    "Marathwada Shikshan Prasarak Mandal's Shri Shivaji Institute of Engineering and Management Studies, Parbhani",
    "Matoshri College of Engineering and Research Centre, Eklahare, Nashik",
    "Matoshri Education Soceity, Matoshri Asarabai Polytechnic,Nashik",
    "Matoshri Education Soceity, Matoshri Institute Of Technology, Dhanore, Nashik",
    "Matoshri Pratishan's Group of Institutions (Integrated Campus), Kupsarwadi , Nanded",
    "Matsyodari Shikshan Sansatha's College of Engineering and Technology, Jalna",
    "Maulana Mukhtar Ahmad Nadvi Technical Campus, Malegaon.",
    "Mauli Group of Institutions, College of Engineering and Technology, Shegaon.",
    "Metropolitan Institute of Technology & Management, Sukhalwad, Sindhudurg.",
    "Modern Education Society's Wadia College of Engineering, Pune",
    "N. B. Navale Sinhgad College of Engineering, Kegaon, solapur",
    "N.Y.S.S.'s Datta Meghe College of Engineering, Airoli, Navi Mumbai",
    "NBN Sinhgad Technical Institutes Campus, Pune",
    "NEW INSTITUTE OF TECHNOLOGY,KOLHAPUR",
    "Nagaon Education Society's Gangamai College of Engineering, Nagaon, Tal Dist Dhule",
    "Nagnathappa Halge Engineering College, Parli, Beed",
    "Nanasaheb Mahadik College of Engineering,Walwa, Sangli.",
    "Navsahyadri Education Society's Group of Institutions",
    "New Horizon Institute of Technology & Management, Thane",
    "Nutan Maharashtra Vidya Prasarak Mandal, Nutan Maharashtra Institute of Engineering &Technology, Talegaon station, Pune",
    "P. R. Pote Patil College of Engineering & Management, Amravati",
    "P.G. College of Engineering & Technology, Nandurbar",
    "P.K. Technical Campus, Pune.",
    "P.S.G.V.P. Mandal's D.N. Patel College of Engineering, Shahada, Dist. Nandurbar",
    "PUNE VIDYARTHI GRIHA'S COLLEGE OF ENGINEERING & SHRIKRUSHNA S. DHAMANKAR INSTITUTE OF MANAGEMENT, NASHIK",
    "Padmashri Dr. V.B. Kolte College of Engineering, Malkapur, Buldhana",
    "Paramhansa Ramkrishna Maunibaba Shikshan Santha's , Anuradha Engineering College, Chikhali",
    "Peoples Education Society's College of Engineering, Aurangabad",
    "Phaltan Education Society's College of Engineering Thakurki Tal- Phaltan Dist-Satara",
    "Pimpri Chinchwad Education Trust's Pimpri Chinchwad College Of Engineering And Research, Ravet",
    "Pimpri Chinchwad Education Trust, Pimpri Chinchwad College of Engineering, Pune",
    "Pradnya Niketan Education Society's Nagesh Karajagi Orchid College of Engineering & Technology, Solapur",
    "Pravara Rural College of Engineering, Loni, Pravaranagar, Ahmednagar.",
    "Pravara Rural Education Society's Sir Visvesvaraya Institute of Technology, Chincholi Dist. Nashik",
    "Pravin Rohidas Patil College of Engineering & Technology",
    "Priyadarshini Bhagwati College of Engineering, Harpur Nagar, Umred Road,Nagpur",
    "Prof Ram Meghe College of Engineering and Management, Badnera",
    "Prof. Ram Meghe Institute of Technology & Research, Amravati",
    "Progressive Education Society's Modern College of Engineering, Pune",
    "Pune District Education Association's College of Engineering, Manjari(Bk), Hadapsar, Pune",
    "Pune Institute of Computer Technology",
    "Pune Vidyarthi Griha's College of Engineering and Technology and G K Pate(Wani) Institute of Management, Pune",
    "R. C. Patel Institute of Technology, Shirpur",
    "R.C. Patel College of Engineering and Polytechnic, Shirpur",
    "Rajendra Mane College of Engineering & Technology Ambav Deorukh",
    "Rajgad Dnyanpeeth's Shri Chhatrapati Shivajiraje College of Engineering, Bhor",
    "Rajiv Gandhi College of Engineering Research & Technology Chandrapur",
    "Rajiv Gandhi College of Engineering, At Post Karjule Hariya Tal.Parner, Dist.Ahmednagar",
    "Rasiklal M. Dhariwal Sinhgad Technical Institutes Campus, Warje, Pune.",
    "Rayat Shikshan Sanstha's Karmaveer Bhaurao Patil College of Engineering, Satara",
    "Rizvi Education Society's Rizvi College of Engineering, Bandra,Mumbai",
    "S K N Sinhgad College of Engineering, Korti Tal. Pandharpur Dist Solapur",
    "S.I.E.S. Graduate School of Technology, Nerul, Navi Mumbai",
    "S.S.P.M.'s College of Engineering, Kankavli",
    "SKN Sinhgad Institute of Technology & Science, Kusgaon(BK),Pune.",
    "SNJB's Late Sau. Kantabai Bhavarlalji Jain College of Engineering, (Jain Gurukul), Neminagar,Chandwad,(Nashik)",
    "ST. Vincent Pallotti College of Engineering & Technology, Nagpur",
    "STMEI's Sandipani Technical Campus-Faculty of Engineering, Latur.",
    "Sahakar Maharshee Shankarrao Mohite Patil Institute of Technology & Research, Akluj",
    "Sahyadri Valley College of Engineering & Technology, Rajuri, Pune.",
    "Samarth College of Engineering and Management",
    "Samarth Education Trust's Arvind Gavali College Of Engineering Panwalewadi, Varye,Satara.",
    "Samridhi Sarwajanik Charitable Trust, Jhulelal Institute of Technology, Nagpur",
    "Sandip Foundation's, Sandip Institute of Engineering & Management, Nashik",
    "Sandip Foundation, Sandip Institute of Technology and Research Centre, Mahiravani, Nashik",
    "Sanghavi College of Engineering, Varvandi, Nashik.",
    "Sanjay Ghodawat Institute",
    "Sanjeevan Group of Institutions",
    "Sanjivani Rural Education Society's Sanjivani College of Engineering, Kopargaon",
    "Sanmarg Shikshan Sanstha's Smt. Radhikatai Pandav College of Engineering, Nagpur",
    "Sanmarg Shikshan Sanstha, Mandukarrao Pandav College of Engineering, Bhandara",
    "Sanmati Engineering College, Sawargaon Barde, Washim",
    "Sant Gadge Baba Amravati University,Amravati",
    "Sant Gajanan Maharaj College of Engineering, Gadhinglaj",
    "Saraswati Education Society's Saraswati College of Engineering,Kharghar Navi Mumbai",
    "Saraswati Education Society, Yadavrao Tasgaonkar Institute of Engineering & Technology, Karjat",
    "Sardar Patel College of Engineering, Andheri",
    "Shahajirao Patil Vikas Pratishthan, S.B.Patil College of Engineering, Vangali, Tal. Indapur",
    "Shanti Education Society, A.G. Patil Institute of Technology, Soregaon, Solapur(North)",
    "Sharad Institute of Technology College of Engineering, Yadrav(Ichalkaranji)",
    "Shetkari Shikshan Mandal's Pad. Vasantraodada Patil Institute of Technology, Budhgaon, Sangli",
    "Shivajirao S. Jondhale College of Engineering, Dombivali,Mumbai",
    "Shivganga Charitable Trust, Sangli Vishveshwarya Technical Campus, Faculty of Diploma Engineering, Patgaon, Miraj",
    "Shivnagar Vidya Prasarak Mandal's College of Engineering, Malegaon-Baramati",
    "Shramsadhana Bombay Trust, College of Engineering & Technology, Jalgaon",
    "Shree Gajanan Maharaj Shikshan Prasarak Manda'l Sharadchandra Pawar College of Engineering, Dumbarwadi",
    "Shree L.R. Tiwari College of Engineering, Mira Road, Mumbai",
    "Shree Ramchandra College of Engineering, Lonikand,Pune",
    "Shree Santkrupa Shikshan Sanstha, Shree Santkrupa Institute Of Engineering & Technology, Karad",
    "Shree Siddheshwar Women's College Of Engineering Solapur.",
    "Shree Tuljabhavani College of Engineering, Tuljapur",
    "Shree Yash Pratishthan, Shreeyash College of Engineering and Technology, Aurangabad",
    "Shri Guru Gobind Singhji Institute of Engineering and Technology, Nanded",
    "Shri Hanuman Vyayam Prasarak Mandals College of Engineering & Technology, Amravati",
    "Shri Sant Gajanan Maharaj College of Engineering,Shegaon",
    "Shri Shivaji Education Society's College of Engineering and Technology, Akola",
    "Shri Shivaji Vidya Prasarak Sanstha's Late Bapusaheb Shivaji Rao Deore College of Engineering,Dhule",
    "Shri Swami Samarth Institute of Management and Technology, Malwadi-Bota",
    "Shri Vile Parle Kelavani Mandal's Institute of Technology, Dhule",
    "Shri Vile Parle Kelvani Mandal's Dwarkadas J. Sanghvi College of Engineering, Vile Parle,Mumbai",
    "Shri Vithal Education and Research Institute's College of Engineering, Pandharpur",
    "Shri. Ambabai Talim Sanstha's Sanjay Bhokare Group of Institutes, Miraj",
    "Shri. Anandrao Abitkar College of Engineering, Pal",
    "Shri. Balasaheb Mane Shikshan Prasarak Mandal's, Ashokrao Mane Group of Institutions",
    "Shri. Jaykumar Rawal Institute of Technology, Dondaicha.",
    "Shri. Sai Shikshan Sanstha, Nagpur Institute of Technology, Nagpur",
    "Shri.Someshwar Shikshan Prasarak Mandal, Sharadchandra Pawar College of Engineering & Technology, Someshwar Nagar",
    "Shriram Gram Vikas Shikshan Sanstha, Vilasrao Deshmukh College of Engineering and Technology, Nagpur",
    "Shriram Institute Of Engineering & Technology, (Poly), Paniv",
    "Siddhant College of Engineering, A/p Sudumbare, Tal.Maval, Dist-Pune",
    "Siddhivinayak Technical Campus, School of Engineering & Research Technology, Shirasgon, Nile",
    "Sinhgad Academy of Engineering, Kondhwa (BK) Kondhwa-Saswad Road, Pune",
    "Sinhgad College of Engineering, Vadgaon (BK), Pune",
    "Sinhgad Institute of Technology",
    "Sinhgad Technical Education Society's Smt. Kashibai Navale College of Engineering,Vadgaon,Pune",
    "Sinhgad Technical Education Society, Sinhgad Institute of Technology and Science, Narhe (Ambegaon)",
    "Sipna Shikshan Prasarak Mandal College of Engineering & Technology, Amravati",
    "Sir Shantilal Badjate Charitable Trust's S. B. Jain Institute of technology, Management & Research, Nagpur",
    "Smt. Indira Gandhi College of Engineering, Navi Mumbai",
    "St. Francis Institute of Technology,Borivali, Mumbai",
    "Suman Ramesh Tulsiani Technical Campus: Faculty of Engineering, Kamshet,Pune.",
    "Suryodaya College of Engineering & Technology, Nagpur",
    "Svkm's Shri Bhagubhai Mafatlal Polytechnic & College of Engineering",
    "Swami Vivekananda Shikshan Sanstha, Dr. Bapuji Salunkhe Institute Of Engineering & Technology,Kolhapur",
    "Swaminarayan Siddhanta Institute Of Technology, Nagpur",
    "T.M.E. Society's J.T.Mahajan College of Engineering, Faizpur",
    "TSSM's Bhivarabai Sawant College of Engineering and Research, Narhe, Pune",
    "TSSMS's Pd. Vasantdada Patil Institute of Technology, Bavdhan, Pune",
    "Takshashila Institute of Engineering & Technology, Darapur, Amravati",
    "Tatyasaheb Kore Institute of Engineering and Technology, Warananagar",
    "Tatyasaheb Kore Institute of Engineering and Technology, Yelur",
    "Terna Engineering College, Nerul, Navi Mumbai",
    "Terna Public Charitable Trust's College of Engineering, Osmanabad",
    "Thadomal Shahani Engineering College, Bandra, Mumbai",
    "Thakur College of Engineering and Technology, Kandivali, Mumbai",
    "Thakur Shyamnarayan Engineering College, Mumbai",
    "Tulsiramji Gaikwad Patil College of Engineering & Technology, Nagpur",
    "Universal College of Engineering,Kaman Dist. Palghar",
    "University Department of Chemical Technology, Aurangabad",
    "University Institute of Chemical Technology, North Maharashtra University, Jalgaon",
    "Usha Mittal Institute of Technology SNDT Women's University, Mumbai",
    "VPM's Maharshi Parshuram College of Engineering, Velneshwar, Ratnagiri.",
    "Vardhaman Education & Welfare Society, Ahinsa Institute of Technology, Post. Dondaicha, Dhule",
    "Vasantdada Patil Pratishthan's College Of Engineering and Visual Arts, Sion, Mumbai",
    "Veermata Jijabai Technological Institute(VJTI), Matunga, Mumbai",
    "Vidya Niketan College of Engineering, Bota Sangamner",
    "Vidya Prasarak Mandal's College of Engineering, Thane",
    "Vidya Prasarini Sabha's College of Engineering & Technology, Lonavala",
    "Vidya Pratishthan's Kamalnayan Bajaj Institute of Engineering & Technology, Baramati Dist.Pune",
    "Vidya Vikas Pratishthan Institute of Engineering and Technology, Solapur",
    "Vidyalankar Institute of Technology,Wadala, Mumbai",
    "Vidyavardhini's College of Engineering and Technology, Vasai",
    "Vighnaharata Trust's Shivajirao S. Jondhale College of Engineering & Technology, Shahapur, Asangaon, Dist Thane",
    "Vilasrao Deshmukh Foundation Group of Institutions, Latur",
    "Vishnu Waman Thakur Charitable Trust's VIVA Institute of Technology, Virar",
    "Vishwabharati Academy's College of Engineering, Ahmednagar",
    "Vishwaniketan's Institute of Management Entrepreneurship and Engineering Technology(i MEET), Khalapur Dist Raigad",
    "Vishwatmak Jangli Maharaj Ashram Trust (Kokamthan), Atma Malik Institute Of Technology & Research",
    "Vision Buldhana Educational & Welfare Society's Pankaj Laddhad Institute of Technology & Management Studies, Yelgaon",
    "Vivekanand Education Society's Institute of Technology, Chembur, Mumbai",
    "Wainganga College of Engineering and Management, Dongargaon, Nagpur",
    "Walchand College of Engineering, Sangli",
    "Walchand Institute of Technology, Solapur",
    "Watumull Institute of Engineering & Technology, Ulhasnagar",
    "Xavier Institute Of Engineering C/O Xavier Technical Institute,Mahim,Mumbai",
    "YASHWANTRAO BHONSALE INSTITUTE OF TECHNOLOGY",
    "Yadavrao Tasgaonkar College of Engineering & Management",
    "Yashoda Technical Campus, Wadhe, Satara.",
    "Yeshwantrao Chavan College of Engineering,Wanadongri, Nagpur",
    "Zeal Education Society's Zeal College of Engineering & Reserch, Narhe, Pune"
]

# ==============================================================================
# 4. STREAMLIT CACHED MODEL LOADERS
# ==============================================================================

INSTITUTE_MODEL_PATH = "institute_model_compressed.pkl.gz"
COURSE_MODEL_PATH = "course_model.pkl"

GENDER_PATH = "gender_encoder.pkl"
CATEGORY_PATH = "category_encoder.pkl"
SEAT_PATH = "seat_encoder.pkl"
COURSE_PATH = "course_encoder.pkl"
INSTITUTE_PATH = "institute_encoder.pkl"

@st.cache_resource
def load_pickle(path):
    if os.path.exists(path):
        if path.endswith(".gz"):
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        else:
            with open(path, "rb") as f:
                return pickle.load(f)
    return None

institute_model = load_pickle(INSTITUTE_MODEL_PATH)
course_model = load_pickle(COURSE_MODEL_PATH)

gender_encoder = load_pickle(GENDER_PATH)
category_encoder = load_pickle(CATEGORY_PATH)
seat_encoder = load_pickle(SEAT_PATH)
course_encoder = load_pickle(COURSE_PATH)
institute_encoder = load_pickle(INSTITUTE_PATH)

def get_classes(encoder, fallback):
    if encoder and hasattr(encoder, 'classes_'):
        return list(encoder.classes_)
    return fallback

GENDER_OPTIONS = get_classes(gender_encoder, FALLBACK_GENDERS)
CATEGORY_OPTIONS = get_classes(category_encoder, FALLBACK_CATEGORIES)
SEAT_OPTIONS = get_classes(seat_encoder, FALLBACK_SEATS)
INSTITUTE_OPTIONS = get_classes(institute_encoder, FALLBACK_INSTITUTES)
COURSE_OPTIONS = get_classes(course_encoder, FALLBACK_COURSES)

if 'history' not in st.session_state:
    st.session_state.history = []

# ==============================================================================
# 5. UI LAYOUT & INTERACTION ENGINE
# ==============================================================================

st.markdown(f"<h1 style='color:{theme['primary']};'>🎓 EduPredict.AI Engine</h1>", unsafe_allow_html=True)
st.caption("Next-Generation Machine Learning Platform for College & Engineering Branch Predictions")

st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📋 Candidate Profile Parameters")
    
    gender_input = st.selectbox("Gender Category", GENDER_OPTIONS)
    category_input = st.selectbox("Reservation Category", CATEGORY_OPTIONS)
    percentile_input = st.number_input(
        "MHTCET Percentile Score", 
        min_value=0.0, 
        max_value=100.0, 
        value=95.0000, 
        step=0.0001,
        format="%.4f"
    )
    seat_input = st.selectbox("Seat Allocation Quota", SEAT_OPTIONS)
    institute_input = st.selectbox("Target Institute Name", INSTITUTE_OPTIONS)

    predict_btn = st.button("Execute Intelligence Inference 🚀", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📊 Allocation Intelligence Dashboard")

    if predict_btn:
        try:
            gender_encoded = int(gender_encoder.transform([gender_input])[0]) if gender_encoder else 0
            category_encoded = int(category_encoder.transform([category_input])[0]) if category_encoder else 0
            seat_encoded = int(seat_encoder.transform([seat_input])[0]) if seat_encoder else 0
            institute_encoded = int(institute_encoder.transform([institute_input])[0]) if institute_encoder else 0

            features = np.array([[gender_encoded, category_encoded, percentile_input, seat_encoded, institute_encoded]])

            active_model = institute_model if institute_model is not None else course_model

            if active_model is not None:
                pred_idx = active_model.predict(features)[0]

                if course_encoder and hasattr(course_encoder, 'inverse_transform'):
                    predicted_course = course_encoder.inverse_transform([pred_idx])[0]
                else:
                    predicted_course = COURSE_OPTIONS[int(pred_idx) % len(COURSE_OPTIONS)]

                st.markdown(f"""
                    <div class="hero-prediction-box">
                        <div class="prediction-title-text">Most Likely Allocated Branch</div>
                        <div class="prediction-value-text">{predicted_course}</div>
                    </div>
                """, unsafe_allow_html=True)

                if hasattr(active_model, "predict_proba"):
                    probs = active_model.predict_proba(features)[0]
                    top_indices = np.argsort(probs)[::-1][:5]

                    st.markdown("#### Confidence Top Matches")
                    for idx in top_indices:
                        if probs[idx] > 0.001:
                            b_name = course_encoder.inverse_transform([idx])[0] if course_encoder else f"Branch {idx}"
                            prob_val = round(float(probs[idx]) * 100, 2)
                            st.write(f"**{b_name}**: `{prob_val}%`")
                            st.progress(float(probs[idx]))

                st.session_state.history.insert(0, {
                    "Institute": institute_input,
                    "Percentile": f"{percentile_input:.4f}%ile",
                    "Category": category_input,
                    "Quota": seat_input,
                    "Predicted Branch": predicted_course
                })
            else:
                st.error("Model files not loaded in application directory.")

        except Exception as e:
            st.error(f"Inference Error: {str(e)}")
    else:
        st.info("Configure parameters on the left and click **Execute Intelligence Inference** to display ML probability models.")
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 6. HISTORICAL LOGS
# ==============================================================================
if st.session_state.history:
    st.divider()
    st.subheader("🕒 Session Historical Predictions")
    df_history = pd.DataFrame(st.session_state.history[:10])
    st.dataframe(df_history, use_container_width=True)
