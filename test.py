# app.py
import os
import streamlit as st
from dotenv import load_dotenv

# =========================================================
# CONFIGURATION & DONNÉES
# =========================================================
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")

# Liste de tickets pour tester
tickets_list = [
    {"id": 1, "description": "Impossible de se connecter au VPN depuis le poste Windows 10.", "status": "open"},
    {"id": 2, "description": "L'imprimante réseau HP ne répond plus depuis la migration VLAN.", "status": "open"},
    {"id": 3, "description": "Erreur 500 sur l'API /orders lors d'une commande.", "status": "open"},
    {"id": 4, "description": "L'application mobile plante au démarrage après la mise à jour.", "status": "open"},
    {"id": 5, "description": "Demande de création d'un compte utilisateur pour le nouvel employé.", "status": "open"},
]

# Tentative d'import de MistralChat depuis Agno
try:
    from agno.models.mistral import MistralChat
except Exception:
    MistralChat = None


# =========================================================
# AGENTS
# =========================================================
class TicketCollectorAgent:
    def __init__(self, tickets):
        self.name = "TicketCollectorAgent"
        self.tickets = tickets

    def collect(self):
        return list(self.tickets)


class NLPAgent:
    def __init__(self, model=None, mock=False):
        self.name = "NLPAgent"
        self.mock = mock
        self.model = model

    def process(self, ticket):
        desc = ticket.get("description", "")
        if self.mock:
            return self._mock_category(desc)
        if not self.model:
            raise RuntimeError("NLPAgent: modèle Mistral non initialisé")
        prompt = [
            {"role": "system", "content": "Tu es un assistant IT qui catégorise des tickets."},
            {"role": "user", "content": f"Catégorise ce ticket en une courte étiquette : {desc}"}
        ]
        return str(self.model.chat(prompt)).strip()

    def _mock_category(self, desc):
        desc_lower = desc.lower()
        if "vpn" in desc_lower or "vlan" in desc_lower or "réseau" in desc_lower:
            return "Réseau"
        if "imprimante" in desc_lower or "hp" in desc_lower:
            return "Matériel"
        if "erreur" in desc_lower or "api" in desc_lower or "500" in desc_lower:
            return "Application"
        if "compte" in desc_lower or "utilisateur" in desc_lower:
            return "Accès"
        return "Général"


class SummarizerAgent:
    def __init__(self, model=None, mock=False):
        self.name = "SummarizerAgent"
        self.mock = mock
        self.model = model

    def process(self, ticket, category):
        desc = ticket.get("description", "")
        if self.mock:
            return desc[:140] + ("..." if len(desc) > 140 else "")
        if not self.model:
            raise RuntimeError("SummarizerAgent: modèle Mistral non initialisé")
        prompt = [
            {"role": "system", "content": "Tu es un assistant qui fait un résumé concis de tickets IT."},
            {"role": "user", "content": f"Résume ce ticket (catégorie: {category}) : {desc}"}
        ]
        return str(self.model.chat(prompt)).strip()


class RecommenderAgent:
    def __init__(self, model=None, mock=False):
        self.name = "RecommenderAgent"
        self.mock = mock
        self.model = model

    def process(self, ticket, summary, category):
        if self.mock:
            return self._mock_recommendation(category)
        if not self.model:
            raise RuntimeError("RecommenderAgent: modèle Mistral non initialisé")
        prompt = [
            {"role": "system", "content": "Tu es un expert IT qui propose des solutions basées sur un résumé de ticket."},
            {"role": "user", "content": f"Propose une solution au ticket résumé suivant (catégorie: {category}) : {summary}"}
        ]
        return str(self.model.chat(prompt)).strip()

    def _mock_recommendation(self, category):
        mapping = {
            "Réseau": "Vérifier la configuration VPN/routeur et redémarrer le service.",
            "Matériel": "Vérifier les câbles et redémarrer l'imprimante.",
            "Application": "Examiner les logs applicatifs et corriger le bug.",
            "Accès": "Créer le compte et attribuer les droits nécessaires.",
            "Général": "Collecter plus d'infos et escalader au support approprié."
        }
        return mapping.get(category, mapping["Général"])


# =========================================================
# PIPELINE
# =========================================================
class ITSupportPipeline:
    def __init__(self, tickets, mock=False, mistral_model_id="mistral-medium", api_key=None):
        self.tickets = tickets
        self.mock = mock
        self.api_key = api_key or API_KEY

        model_instance = None
        if not self.mock:
            if MistralChat is None:
                raise ImportError("MistralChat introuvable. Installe agno ou passe mock=True.")
            if not self.api_key:
                raise ValueError("MISTRAL_API_KEY manquant. Ajoute-le dans .env ou passe mock=True.")
            model_instance = MistralChat(id=mistral_model_id, api_key=self.api_key)

        self.collector = TicketCollectorAgent(self.tickets)
        self.nlp = NLPAgent(model=model_instance, mock=self.mock)
        self.summarizer = SummarizerAgent(model=model_instance, mock=self.mock)
        self.recommender = RecommenderAgent(model=model_instance, mock=self.mock)

    def analyze_ticket(self, ticket):
        category = self.nlp.process(ticket)
        summary = self.summarizer.process(ticket, category)
        recommendation = self.recommender.process(ticket, summary, category)
        return {
            "id": ticket["id"],
            "description": ticket["description"],
            "status": ticket["status"],
            "category": category,
            "summary": summary,
            "recommendation": recommendation
        }

    def analyze_all(self):
        return [self.analyze_ticket(t) for t in self.collector.collect()]


# =========================================================
# FRONTEND STREAMLIT
# =========================================================
st.set_page_config(page_title="IT Support Analyzer", layout="centered")
st.title("🎯 Analyse de tickets IT — Multi-Agent (Agno + Mistral)")

mock_mode = st.checkbox("Mode mock (réponses simulées, pas d'appel API)", value=True)

with st.sidebar:
    st.header("Paramètres")
    st.write("Mode mock = pas besoin de clé API")
    run_all = st.button("Analyser tous les tickets")
    st.write("Tickets disponibles :", len(tickets_list))

ticket_ids = [t["id"] for t in tickets_list]
selected_id = st.selectbox("Choisir un ticket", ticket_ids)
ticket = next(t for t in tickets_list if t["id"] == selected_id)

st.subheader(f"Ticket #{ticket['id']}")
desc = st.text_area("Description du ticket :", value=ticket["description"], height=120)

if st.button("Analyser ce ticket"):
    pipeline = ITSupportPipeline(tickets=[{**ticket, "description": desc}], mock=mock_mode)
    with st.spinner("Analyse en cours..."):
        res = pipeline.analyze_ticket({**ticket, "description": desc})
    st.success("✅ Analyse terminée")
    st.write("**Catégorie :**", res["category"])
    st.write("**Résumé :**", res["summary"])
    st.write("**Recommandation :**", res["recommendation"])

if run_all:
    pipeline_all = ITSupportPipeline(tickets=tickets_list, mock=mock_mode)
    with st.spinner("Analyse de tous les tickets..."):
        results = pipeline_all.analyze_all()
    st.success(f"✅ {len(results)} tickets analysés")
    for r in results:
        st.markdown(f"---\n**Ticket #{r['id']}** — *{r['status']}*")
        st.write("Description :", r["description"])
        st.write("Catégorie :", r["category"])
        st.write("Résumé :", r["summary"])
        st.write("Recommandation :", r["recommendation"])
