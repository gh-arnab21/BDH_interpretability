#!/usr/bin/env python3
"""
Pre-compute monosemanticity data for static frontend visualization.

V2 REWRITE — addresses fundamental issues:
1. Sentence-level extraction at concept-word positions (not single-word mean-pool)
2. Position-level selectivity from diverse sentence corpus
3. Contrastive cross-concept with contextualized fingerprints
4. Proper French words for all concepts (model is French-trained)
5. sigma synapse tracking with concept-selective delta-sigma discovery

Generates a single JSON file consumed by the frontend.

Usage:
    python scripts/precompute_monosemanticity.py \
        --model checkpoints/french/french_best.pt \
        --output frontend/public/monosemanticity/precomputed.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch

# ── Resolve imports ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from bdh import BDH, BDHConfig, ExtractionConfig, load_model  # noqa: E402

# ══════════════════════════════════════════════════════════════════════
#  CURATED CATEGORIES — all words in French (model trained on FR text)
# ══════════════════════════════════════════════════════════════════════
CATEGORIES: Dict[str, Dict[str, Any]] = {
    "currencies": {
        "name": "Currencies",
        "icon": "\U0001f4b0",
        "words": ["dollar", "euro", "franc", "yen"],
    },
    "countries": {
        "name": "Countries",
        "icon": "\U0001f30d",
        "words": ["france", "allemagne", "espagne", "italie"],
    },
    "languages": {
        "name": "Languages",
        "icon": "\U0001f5e3\ufe0f",
        "words": ["anglais", "fran\u00e7ais", "espagnol", "allemand"],
    },
    "politics": {
        "name": "Politics",
        "icon": "\u2696\ufe0f",
        "words": ["parlement", "commission", "conseil", "vote"],
    },
}

# Cross-concept pairs to pre-compute
CROSS_PAIRS = [
    ("currencies", "countries"),
    ("currencies", "languages"),
    ("countries", "politics"),
    ("languages", "politics"),
]

# ══════════════════════════════════════════════════════════════════════
#  SENTENCE CORPUS — diverse French sentences with concept words
#  Each sentence embeds concept words in natural EU parliament / news
#  context so the model can build contextual representations.
#  ~13 sentences per concept word, ~52 per concept category.
# ══════════════════════════════════════════════════════════════════════
SENTENCE_CORPUS: Dict[str, List[str]] = {
    "currencies": [
        # dollar (13 sentences)
        "le dollar am\u00e9ricain reste la monnaie de r\u00e9f\u00e9rence mondiale",
        "le cours du dollar a fortement augment\u00e9 cette semaine",
        "la valeur du dollar influence les march\u00e9s internationaux",
        "il faut convertir les prix en dollar pour comparer",
        "le budget est exprim\u00e9 en dollar et en euro",
        "les r\u00e9serves mondiales sont principalement en dollar",
        "la dette publique est libell\u00e9e en dollar dans plusieurs pays",
        "le dollar a perdu du terrain face aux autres devises",
        "les mati\u00e8res premi\u00e8res sont cot\u00e9es en dollar sur les march\u00e9s",
        "la politique am\u00e9ricaine a un impact direct sur le dollar",
        "le commerce international d\u00e9pend largement du dollar",
        "les pays exportateurs de p\u00e9trole vendent en dollar",
        "la chute du dollar a provoqu\u00e9 une crise de confiance",
        # euro (13 sentences)
        "la zone euro comprend dix-neuf pays europ\u00e9ens",
        "le taux de change de l'euro reste relativement stable",
        "le budget europ\u00e9en est enti\u00e8rement calcul\u00e9 en euro",
        "un euro vaut environ un dollar vingt aujourd'hui",
        "la stabilit\u00e9 de l'euro d\u00e9pend de la politique mon\u00e9taire",
        "la banque centrale europ\u00e9enne g\u00e8re la politique de l'euro",
        "les citoyens de la zone euro utilisent une monnaie commune",
        "le passage \u00e0 l'euro a transform\u00e9 les \u00e9changes commerciaux",
        "la valeur de l'euro a augment\u00e9 par rapport au yen",
        "les obligations d'\u00e9tat sont \u00e9mises en euro dans la zone",
        "le march\u00e9 unique fonctionne gr\u00e2ce \u00e0 l'euro",
        "les transactions financi\u00e8res en euro sont en forte croissance",
        "la force de l'euro renforce la position europ\u00e9enne",
        # franc (13 sentences)
        "le franc suisse est consid\u00e9r\u00e9 comme une valeur refuge",
        "le franc a \u00e9t\u00e9 la monnaie fran\u00e7aise pendant des si\u00e8cles",
        "la conversion du franc en euro a eu lieu en deux mille deux",
        "le cours du franc reste stable face aux autres devises",
        "le franc belge a \u00e9galement \u00e9t\u00e9 remplac\u00e9 par l'euro",
        "la nostalgie du franc persiste chez certains citoyens",
        "le franc africain est utilis\u00e9 dans plusieurs pays du continent",
        "les prix \u00e9taient affich\u00e9s en franc avant la transition",
        "le franc suisse attire les investisseurs en p\u00e9riode de crise",
        "la parit\u00e9 du franc avec l'euro a \u00e9t\u00e9 fix\u00e9e d\u00e9finitivement",
        "les salaires \u00e9taient vers\u00e9s en franc avant deux mille deux",
        "le franc a jou\u00e9 un r\u00f4le central dans l'\u00e9conomie fran\u00e7aise",
        "la valeur du franc variait selon les accords mon\u00e9taires",
        # yen (13 sentences)
        "le yen japonais a atteint un nouveau record historique",
        "les investisseurs internationaux se tournent vers le yen",
        "le cours du yen reste tr\u00e8s volatil cette ann\u00e9e",
        "la banque centrale contr\u00f4le la valeur du yen",
        "le yen est la troisi\u00e8me monnaie la plus \u00e9chang\u00e9e au monde",
        "la faiblesse du yen favorise les exportations japonaises",
        "les touristes europ\u00e9ens profitent du taux favorable du yen",
        "le yen a subi une d\u00e9pr\u00e9ciation significative cette ann\u00e9e",
        "les r\u00e9serves de change incluent une part importante en yen",
        "le rapport entre l'euro et le yen fluctue consid\u00e9rablement",
        "la hausse du yen inqui\u00e8te les march\u00e9s asiatiques",
        "les obligations japonaises sont libell\u00e9es en yen",
        "le yen reste un indicateur cl\u00e9 de l'\u00e9conomie mondiale",
    ],
    "countries": [
        # france (13 sentences)
        "la france est un membre fondateur de l'union europ\u00e9enne",
        "le pr\u00e9sident de la france a prononc\u00e9 un discours important",
        "la france a soutenu cette r\u00e9solution au conseil europ\u00e9en",
        "les exportations de la france ont augment\u00e9 ce trimestre",
        "la france participe activement aux n\u00e9gociations commerciales",
        "la france et l'allemagne forment le moteur de l'europe",
        "le syst\u00e8me \u00e9ducatif de la france est reconnu mondialement",
        "la france accueille chaque ann\u00e9e des millions de touristes",
        "la recherche scientifique en france b\u00e9n\u00e9ficie de fonds europ\u00e9ens",
        "la france a propos\u00e9 un plan ambitieux pour le climat",
        "le secteur agricole de la france est le plus grand d'europe",
        "la france d\u00e9fend une politique \u00e9trang\u00e8re ind\u00e9pendante",
        "la contribution de la france au budget europ\u00e9en est significative",
        # allemagne (13 sentences)
        "l'allemagne est la premi\u00e8re \u00e9conomie du continent europ\u00e9en",
        "le gouvernement de l'allemagne propose un nouveau budget",
        "l'allemagne a vot\u00e9 en faveur de cette directive",
        "les industries de l'allemagne exportent dans le monde entier",
        "l'allemagne investit massivement dans les \u00e9nergies renouvelables",
        "le march\u00e9 du travail en allemagne reste dynamique et comp\u00e9titif",
        "l'allemagne accueille de nombreux travailleurs europ\u00e9ens qualifi\u00e9s",
        "la politique \u00e9nerg\u00e9tique de l'allemagne est en pleine transition",
        "l'allemagne soutient la coop\u00e9ration bilat\u00e9rale avec ses voisins",
        "les universit\u00e9s de l'allemagne attirent des \u00e9tudiants du monde entier",
        "l'allemagne joue un r\u00f4le majeur dans les institutions europ\u00e9ennes",
        "la r\u00e9unification de l'allemagne a transform\u00e9 l'europe moderne",
        "l'allemagne milite pour une politique budg\u00e9taire europ\u00e9enne stricte",
        # espagne (13 sentences)
        "l'espagne a rejoint l'union europ\u00e9enne en mille neuf cent quatre-vingt-six",
        "le tourisme en espagne contribue fortement \u00e0 son \u00e9conomie",
        "l'espagne a adopt\u00e9 des r\u00e9formes \u00e9conomiques importantes",
        "la croissance de l'espagne d\u00e9passe la moyenne europ\u00e9enne",
        "l'espagne poss\u00e8de un riche patrimoine culturel et historique",
        "les r\u00e9gions autonomes de l'espagne ont des comp\u00e9tences \u00e9tendues",
        "l'agriculture en espagne b\u00e9n\u00e9ficie des aides europ\u00e9ennes",
        "l'espagne est un partenaire commercial essentiel pour la france",
        "le ch\u00f4mage en espagne a significativement diminu\u00e9 ces derni\u00e8res ann\u00e9es",
        "l'espagne investit dans les infrastructures de transport modernes",
        "les \u00e9nergies solaires en espagne repr\u00e9sentent un secteur en expansion",
        "l'espagne participe activement aux missions europ\u00e9ennes de d\u00e9fense",
        "la politique migratoire de l'espagne fait l'objet de d\u00e9bats",
        # italie (13 sentences)
        "l'italie est connue pour son patrimoine culturel exceptionnel",
        "le gouvernement de l'italie fait face \u00e0 des d\u00e9fis \u00e9conomiques",
        "l'italie a ratifi\u00e9 le trait\u00e9 de lisbonne rapidement",
        "les r\u00e9gions du nord de l'italie sont tr\u00e8s industrialis\u00e9es",
        "l'italie d\u00e9fend la politique agricole commune au sein de l'europe",
        "le secteur textile de l'italie est r\u00e9put\u00e9 dans le monde entier",
        "l'italie accueille de nombreux sommets internationaux chaque ann\u00e9e",
        "la dette publique de l'italie pr\u00e9occupe les march\u00e9s financiers",
        "l'italie milite pour une r\u00e9forme du syst\u00e8me d'asile europ\u00e9en",
        "les exportations alimentaires de l'italie sont en constante augmentation",
        "l'italie et la france partagent des fronti\u00e8res et des int\u00e9r\u00eats communs",
        "le tourisme culturel en italie attire des visiteurs du monde entier",
        "l'italie contribue activement aux programmes spatiaux europ\u00e9ens",
    ],
    "languages": [
        # anglais (13 sentences)
        "les documents officiels sont disponibles en anglais et en fran\u00e7ais",
        "la traduction en anglais est obligatoire pour les textes europ\u00e9ens",
        "il parle couramment anglais depuis son enfance",
        "les d\u00e9bats sont souvent men\u00e9s en anglais au parlement",
        "l'anglais est la langue la plus utilis\u00e9e dans les institutions",
        "la ma\u00eetrise de l'anglais est un atout professionnel majeur",
        "les conf\u00e9rences scientifiques se tiennent principalement en anglais",
        "l'enseignement de l'anglais commence d\u00e8s l'\u00e9cole primaire",
        "les n\u00e9gociations commerciales se d\u00e9roulent souvent en anglais",
        "les publications acad\u00e9miques sont majoritairement r\u00e9dig\u00e9es en anglais",
        "l'anglais sert de lingua franca dans les organisations internationales",
        "la domination de l'anglais dans les m\u00e9dias est un sujet de d\u00e9bat",
        "les start-ups europ\u00e9ennes communiquent souvent en anglais",
        # fran\u00e7ais (13 sentences)
        "le fran\u00e7ais est une langue officielle de l'union europ\u00e9enne",
        "les discours sont prononc\u00e9s en fran\u00e7ais lors des s\u00e9ances",
        "apprendre le fran\u00e7ais est populaire dans de nombreux pays",
        "la version en fran\u00e7ais du rapport est maintenant disponible",
        "le fran\u00e7ais reste la langue diplomatique par excellence",
        "l'enseignement du fran\u00e7ais se d\u00e9veloppe rapidement en afrique",
        "le fran\u00e7ais est parl\u00e9 sur les cinq continents du globe",
        "la richesse litt\u00e9raire du fran\u00e7ais est reconnue mondialement",
        "les institutions de la francophonie promeuvent le fran\u00e7ais",
        "le fran\u00e7ais occupe une place importante aux nations unies",
        "la d\u00e9fense du fran\u00e7ais face \u00e0 l'anglais mobilise les intellectuels",
        "les \u00e9tudiants \u00e9trangers choisissent souvent d'apprendre le fran\u00e7ais",
        "le fran\u00e7ais juridique est utilis\u00e9 dans les cours europ\u00e9ennes",
        # espagnol (13 sentences)
        "l'espagnol est parl\u00e9 dans de nombreux pays du monde",
        "la traduction en espagnol sera disponible prochainement",
        "les communaut\u00e9s qui parlent espagnol sont tr\u00e8s diverses",
        "il ma\u00eetrise parfaitement l'espagnol et le portugais",
        "l'espagnol est la deuxi\u00e8me langue maternelle la plus r\u00e9pandue",
        "les cours en espagnol attirent de plus en plus d'\u00e9tudiants",
        "la litt\u00e9rature en espagnol a produit de grands auteurs",
        "l'espagnol est une langue officielle aux nations unies",
        "les m\u00e9dias en espagnol touchent un public de plusieurs centaines de millions",
        "apprendre l'espagnol ouvre des portes en am\u00e9rique latine",
        "l'espagnol et le portugais partagent de nombreuses similitudes",
        "les \u00e9changes culturels en espagnol enrichissent la diversit\u00e9 europ\u00e9enne",
        "la demande pour des traducteurs en espagnol ne cesse de cro\u00eetre",
        # allemand (13 sentences)
        "l'allemand est la langue maternelle la plus parl\u00e9e en europe",
        "les textes juridiques sont traduits en allemand syst\u00e9matiquement",
        "parler allemand est un atout sur le march\u00e9 du travail",
        "la version en allemand du document sera publi\u00e9e demain",
        "l'allemand est enseign\u00e9 dans de nombreuses \u00e9coles europ\u00e9ennes",
        "les publications techniques sont souvent disponibles en allemand",
        "la philosophie et la science comptent de grands textes en allemand",
        "l'allemand est une langue germanique parl\u00e9e aussi en autriche",
        "les touristes apprennent quelques mots en allemand avant de voyager",
        "la traduction en allemand des directives est une obligation l\u00e9gale",
        "l'allemand technique est indispensable dans l'industrie automobile",
        "les programmes d'\u00e9change encouragent l'apprentissage de l'allemand",
        "la connaissance de l'allemand facilite les relations commerciales",
    ],
    "politics": [
        # parlement (13 sentences)
        "le parlement europ\u00e9en a vot\u00e9 cette r\u00e9solution ce matin",
        "les d\u00e9put\u00e9s du parlement ont d\u00e9battu pendant des heures",
        "le parlement si\u00e8ge alternativement \u00e0 strasbourg et bruxelles",
        "la session du parlement a \u00e9t\u00e9 particuli\u00e8rement mouvement\u00e9e",
        "le r\u00f4le du parlement est de repr\u00e9senter les citoyens",
        "le parlement a adopt\u00e9 un amendement sur la politique agricole",
        "la commission pr\u00e9sente son rapport annuel devant le parlement",
        "le pr\u00e9sident du parlement a ouvert la s\u00e9ance pl\u00e9ni\u00e8re",
        "les groupes politiques du parlement pr\u00e9parent leurs propositions",
        "le parlement contr\u00f4le l'utilisation du budget communautaire",
        "les auditions du parlement permettent un examen approfondi",
        "le parlement d\u00e9bat des priorit\u00e9s l\u00e9gislatives de la session",
        "la transparence du parlement s'am\u00e9liore gr\u00e2ce aux nouvelles r\u00e8gles",
        # commission (13 sentences)
        "la commission europ\u00e9enne propose un nouveau r\u00e8glement ambitieux",
        "le pr\u00e9sident de la commission a pr\u00e9sent\u00e9 son programme",
        "la commission travaille sur une directive environnementale",
        "les membres de la commission se r\u00e9unissent chaque semaine",
        "la commission a lanc\u00e9 une consultation publique sur le sujet",
        "le rapport de la commission sera publi\u00e9 le mois prochain",
        "la commission surveille le respect des trait\u00e9s par les \u00e9tats membres",
        "la commission n\u00e9gocie les accords commerciaux au nom de l'union",
        "les priorit\u00e9s de la commission incluent le num\u00e9rique et le climat",
        "la commission dispose du pouvoir d'initiative l\u00e9gislative",
        "le coll\u00e8ge des commissaires de la commission vote les propositions",
        "la commission europ\u00e9enne emploie des milliers de fonctionnaires",
        "les d\u00e9cisions de la commission affectent directement les citoyens",
        # conseil (13 sentences)
        "le conseil de l'union europ\u00e9enne a adopt\u00e9 cette position",
        "la pr\u00e9sidence du conseil change tous les six mois",
        "le conseil a approuv\u00e9 le budget pour l'ann\u00e9e prochaine",
        "les ministres du conseil d\u00e9battent des politiques communes",
        "le conseil europ\u00e9en fixe les orientations politiques g\u00e9n\u00e9rales",
        "les r\u00e9unions du conseil se tiennent r\u00e9guli\u00e8rement \u00e0 bruxelles",
        "le conseil statue \u00e0 la majorit\u00e9 qualifi\u00e9e sur la plupart des sujets",
        "la position commune du conseil a \u00e9t\u00e9 transmise au parlement",
        "le conseil a adopt\u00e9 des sanctions contre certains pays tiers",
        "les travaux pr\u00e9paratoires du conseil sont men\u00e9s par le coreper",
        "le conseil et le parlement col\u00e9gif\u00e8rent sur la majorit\u00e9 des textes",
        "le conseil des affaires \u00e9trang\u00e8res coordonne la politique ext\u00e9rieure",
        "les conclusions du conseil orientent les politiques des \u00e9tats membres",
        # vote (13 sentences)
        "le vote sur cette proposition a \u00e9t\u00e9 report\u00e9 \u00e0 demain",
        "chaque d\u00e9put\u00e9 dispose d'un seul vote lors du scrutin",
        "le r\u00e9sultat du vote a \u00e9t\u00e9 annonc\u00e9 en s\u00e9ance pl\u00e9ni\u00e8re",
        "le vote \u00e9lectronique permet un d\u00e9compte rapide et pr\u00e9cis",
        "le vote par appel nominal est demand\u00e9 par cinquante d\u00e9put\u00e9s",
        "la proc\u00e9dure de vote exige une majorit\u00e9 absolue des membres",
        "les abstentions lors du vote refl\u00e8tent des divisions internes",
        "le vote final sur le budget a lieu en d\u00e9cembre",
        "les d\u00e9put\u00e9s expriment leur position par un vote solennel",
        "le vote a confirm\u00e9 le soutien du parlement \u00e0 cette initiative",
        "chaque \u00e9tat membre dispose d'un nombre de vote pond\u00e9r\u00e9",
        "le vote de d\u00e9fiance peut renverser la commission",
        "la majorit\u00e9 n\u00e9cessaire pour le vote d\u00e9pend du type de proc\u00e9dure",
    ],
}

# Sentences for synapse tracking (diverse contexts, 8 sentences per concept)
TRACKING_SENTENCES: Dict[str, List[str]] = {
    "currencies": [
        "le dollar est la monnaie officielle des \u00e9tats-unis",
        "un euro vaut environ un dollar vingt aujourd'hui",
        "le parlement a vot\u00e9 pour le budget en euro",
        "le franc suisse reste tr\u00e8s stable cette ann\u00e9e",
        "les march\u00e9s internationaux surveillent le cours du yen",
        "la politique mon\u00e9taire affecte directement la valeur du dollar",
        "les obligations en euro offrent un rendement int\u00e9ressant",
        "la conversion du franc en euro a simplifi\u00e9 les \u00e9changes",
    ],
    "countries": [
        "la france est un pays europ\u00e9en tr\u00e8s important",
        "le gouvernement de l'allemagne propose un budget ambitieux",
        "le tourisme en espagne repr\u00e9sente une part majeure",
        "les r\u00e9gions du nord de l'italie sont industrialis\u00e9es",
        "la france et l'allemagne coop\u00e8rent dans de nombreux domaines",
        "l'espagne a adopt\u00e9 des r\u00e9formes \u00e9conomiques significatives",
        "l'italie contribue activement aux projets europ\u00e9ens",
        "la france d\u00e9fend sa position sur la politique agricole commune",
    ],
    "languages": [
        "il parle couramment fran\u00e7ais et anglais depuis toujours",
        "l'espagnol est la deuxi\u00e8me langue la plus parl\u00e9e au monde",
        "les textes juridiques sont traduits en allemand et en fran\u00e7ais",
        "l'anglais reste la langue principale des \u00e9changes commerciaux",
        "apprendre le fran\u00e7ais est tr\u00e8s populaire en afrique",
        "l'allemand technique est essentiel dans l'industrie europ\u00e9enne",
        "les publications scientifiques sont lu en anglais partout",
        "la traduction en espagnol des textes officiels est obligatoire",
    ],
    "politics": [
        "le parlement europ\u00e9en a vot\u00e9 cette r\u00e9solution ce matin",
        "la commission propose un nouveau r\u00e8glement tr\u00e8s ambitieux",
        "le conseil a adopt\u00e9 cette r\u00e9solution \u00e0 l'unanimit\u00e9",
        "le r\u00e9sultat du vote a surpris tous les observateurs",
        "le parlement et la commission travaillent ensemble sur ce texte",
        "les d\u00e9put\u00e9s du parlement d\u00e9battent des priorit\u00e9s l\u00e9gislatives",
        "le conseil europ\u00e9en fixe les grandes orientations politiques",
        "la proc\u00e9dure de vote exige une majorit\u00e9 absolue des membres",
    ],
}


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def _split_sentence_to_words(sentence: str) -> List[Tuple[str, int, int]]:
    """
    Split a sentence into words with their byte-range positions.
    Returns: [(word, byte_start, byte_end), ...]
    """
    words: List[Tuple[str, int, int]] = []
    byte_pos = 0
    for word in sentence.split(" "):
        word_bytes = len(word.encode("utf-8"))
        words.append((word, byte_pos, byte_pos + word_bytes))
        byte_pos += word_bytes + 1  # +1 for the space byte
    return words


def _is_concept_word(token: str, concept_words_set: set) -> bool:
    """Check if a token matches a concept word, handling l'/d' prefixes."""
    clean = token.strip(".,;:!?'\"()").lower()
    if clean in concept_words_set:
        return True
    if clean.startswith("l'") and clean[2:] in concept_words_set:
        return True
    if clean.startswith("d'") and clean[2:] in concept_words_set:
        return True
    return False


def _find_word_in_sentence(sentence: str, target_word: str) -> Optional[Tuple[int, int]]:
    """
    Find the byte range of a target word in a sentence.
    Returns (byte_start, byte_end) or None if not found.
    Handles French articles: l', d'.
    """
    words_with_positions = _split_sentence_to_words(sentence.lower())
    target_lower = target_word.lower()

    for word, byte_start, byte_end in words_with_positions:
        clean_word = word.strip(".,;:!?'\"()").lower()
        if clean_word == target_lower:
            return (byte_start, byte_end)
        if clean_word.startswith("l'") and clean_word[2:] == target_lower:
            prefix_bytes = len("l'".encode("utf-8"))
            return (byte_start + prefix_bytes, byte_end)
        if clean_word.startswith("d'") and clean_word[2:] == target_lower:
            prefix_bytes = len("d'".encode("utf-8"))
            return (byte_start + prefix_bytes, byte_end)
    return None


# ══════════════════════════════════════════════════════════════════════
#  PHASE 1: SENTENCE-LEVEL FINGERPRINTING
# ══════════════════════════════════════════════════════════════════════

def extract_fingerprint(
    model: BDH,
    concept_words: List[str],
    sentences: List[str],
    concept_name: str,
    device: str,
    n_layers: int,
    n_heads: int,
    n_neurons: int,
) -> Dict[str, Any]:
    """
    Extract contextualized fingerprints by running full sentences and
    extracting x_sparse at the byte-position of each concept word.

    For each concept word:
    1. Find all sentences containing that word
    2. Run each sentence through the model
    3. Extract x_sparse at the byte positions of the concept word
    4. Average across all sentence occurrences

    This produces CONTEXTUALIZED fingerprints where neuron activations
    reflect the word's meaning in sentence context, not just its byte pattern.
    """
    word_fingerprints: List[Dict] = []
    raw_x: Dict[int, Dict[int, list]] = {}

    for word in concept_words:
        matching_sentences = [s for s in sentences if _find_word_in_sentence(s, word) is not None]

        if not matching_sentences:
            print(f"     \u26a0 No sentences for '{word}' \u2014 using single word")
            matching_sentences = [word]

        # Accumulate x_sparse across sentences: layer -> head -> (total_vec, count)
        accum_x: Dict[int, Dict[int, Tuple[np.ndarray, int]]] = {}

        for sent in matching_sentences:
            tokens = torch.tensor(
                [list(sent.encode("utf-8"))],
                dtype=torch.long,
                device=device,
            )
            word_pos = _find_word_in_sentence(sent, word)
            if word_pos is None:
                continue
            byte_start, byte_end = word_pos

            extraction_config = ExtractionConfig(
                capture_sparse_activations=True,
                capture_attention_patterns=False,
            )

            with torch.no_grad():
                with model.extraction_mode(extraction_config) as buffer:
                    model(tokens)

                    for layer_idx in sorted(buffer.x_sparse.keys()):
                        x = buffer.x_sparse[layer_idx][0]  # (nh, T, N)
                        T = x.shape[1]

                        if layer_idx not in accum_x:
                            accum_x[layer_idx] = {}

                        for h in range(n_heads):
                            # Average over bytes comprising the target word
                            word_vecs = []
                            for pos in range(byte_start, min(byte_end, T)):
                                word_vecs.append(x[h, pos].cpu().numpy())
                            if word_vecs:
                                avg_vec = np.mean(word_vecs, axis=0)
                                if h not in accum_x[layer_idx]:
                                    accum_x[layer_idx][h] = (avg_vec.copy(), 1)
                                else:
                                    prev, cnt = accum_x[layer_idx][h]
                                    accum_x[layer_idx][h] = (prev + avg_vec, cnt + 1)

        # Build fingerprint per word
        layers_data: List[Dict] = []
        for layer_idx in sorted(accum_x.keys()):
            heads_data: List[Dict] = []
            for h in range(n_heads):
                if h in accum_x[layer_idx]:
                    total_vec, cnt = accum_x[layer_idx][h]
                    x_mean = total_vec / cnt
                else:
                    x_mean = np.zeros(n_neurons)

                bins = 64
                stride = max(1, n_neurons // bins)
                x_ds = []
                for b in range(bins):
                    start = b * stride
                    end = min(start + stride, n_neurons)
                    x_ds.append(float(x_mean[start:end].max()))

                x_active = int((x_mean > 0).sum())

                top_k = 20
                top_idx = np.argsort(x_mean)[-top_k:][::-1]
                top_neurons = [
                    {"idx": int(i), "val": round(float(x_mean[i]), 5)}
                    for i in top_idx if x_mean[i] > 0
                ]

                heads_data.append({
                    "head": h,
                    "x_ds": x_ds,
                    "x_active": x_active,
                    "top_neurons": top_neurons,
                })

                raw_x.setdefault(layer_idx, {}).setdefault(h, []).append(x_mean)

            layers_data.append({"layer": layer_idx, "heads": heads_data})

        word_fingerprints.append({"word": word, "layers": layers_data})

    # ── Cosine similarity (per layer, avg across heads) ──
    n_words = len(concept_words)
    similarity_by_layer: Dict[str, list] = {}

    for layer_idx in sorted(raw_x.keys()):
        sim_matrix = np.zeros((n_words, n_words))
        for h in range(n_heads):
            vecs = np.stack(raw_x[layer_idx][h])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
            normed = vecs / norms
            cos = normed @ normed.T
            sim_matrix += cos
        sim_matrix /= n_heads
        similarity_by_layer[str(layer_idx)] = [
            [round(float(sim_matrix[i][j]), 4) for j in range(n_words)]
            for i in range(n_words)
        ]

    # ── Shared neurons ──
    shared_neurons: List[Dict] = []
    for layer_idx in sorted(raw_x.keys()):
        for h in range(n_heads):
            acts = np.stack(raw_x[layer_idx][h])
            active_mask = acts > 0
            all_active = active_mask.all(axis=0)
            shared_idx = np.where(all_active)[0]

            if len(shared_idx) > 0:
                mean_vals = acts[:, shared_idx].mean(axis=0)
                sort_order = np.argsort(mean_vals)[::-1][:5]
                for rank in sort_order:
                    nidx = int(shared_idx[rank])
                    shared_neurons.append({
                        "layer": int(layer_idx),
                        "head": int(h),
                        "neuron": nidx,
                        "mean_activation": round(float(mean_vals[rank]), 5),
                        "active_in": n_words,
                        "per_word": [round(float(acts[w, nidx]), 5) for w in range(n_words)],
                    })

    shared_neurons.sort(key=lambda s: s["mean_activation"], reverse=True)

    return {
        "concept": concept_name,
        "words": word_fingerprints,
        "similarity": similarity_by_layer,
        "shared_neurons": shared_neurons[:40],
        "model_info": {"n_layers": n_layers, "n_heads": n_heads, "n_neurons": n_neurons},
        "raw_x": raw_x,
    }


# ══════════════════════════════════════════════════════════════════════
#  PHASE 2: BEST LAYER
# ══════════════════════════════════════════════════════════════════════

def compute_best_layer(concepts: Dict[str, Any], n_layers: int) -> int:
    """Find the layer with highest avg within-concept cosine similarity."""
    layer_scores: Dict[int, List[float]] = {l: [] for l in range(n_layers)}
    for _cid, result in concepts.items():
        sim = result["similarity"]
        for layer_str, matrix in sim.items():
            layer_idx = int(layer_str)
            n = len(matrix)
            total = sum(matrix[i][j] for i in range(n) for j in range(n) if i != j)
            count = n * (n - 1) if n > 1 else 1
            layer_scores[layer_idx].append(total / count)
    avg_scores = {l: np.mean(v) for l, v in layer_scores.items() if v}
    return int(max(avg_scores, key=avg_scores.get))


# ══════════════════════════════════════════════════════════════════════
#  PHASE 3: CROSS-CONCEPT DISTINCTNESS
# ══════════════════════════════════════════════════════════════════════

def compute_cross_concept(
    concepts: Dict[str, Any], n_layers: int, n_heads: int
) -> List[Dict]:
    """
    Per-layer Jaccard distinctness between top neuron sets of two concepts.
    Now uses CONTEXTUALIZED fingerprints.
    """
    cross_results: List[Dict] = []
    for primary_id, secondary_id in CROSS_PAIRS:
        p_result = concepts.get(primary_id)
        s_result = concepts.get(secondary_id)
        if not p_result or not s_result:
            continue

        distinctness_per_layer: List[float] = []
        for l in range(n_layers):
            p_neurons = set()
            for w in p_result["words"]:
                layer = next((la for la in w["layers"] if la["layer"] == l), None)
                if layer:
                    for h_data in layer["heads"]:
                        for n in h_data["top_neurons"]:
                            p_neurons.add(f"{h_data['head']}_{n['idx']}")

            s_neurons = set()
            for w in s_result["words"]:
                layer = next((la for la in w["layers"] if la["layer"] == l), None)
                if layer:
                    for h_data in layer["heads"]:
                        for n in h_data["top_neurons"]:
                            s_neurons.add(f"{h_data['head']}_{n['idx']}")

            intersection = len(p_neurons & s_neurons)
            union_size = len(p_neurons | s_neurons)
            distinctness_per_layer.append(
                round(1 - intersection / union_size, 4) if union_size > 0 else 1.0
            )

        s_clean = {k: v for k, v in s_result.items() if k != "raw_x"}
        cross_results.append({
            "primary": primary_id,
            "secondary": secondary_id,
            "distinctness_per_layer": distinctness_per_layer,
            "secondary_result": s_clean,
        })

    return cross_results


# ══════════════════════════════════════════════════════════════════════
#  PHASE 4: POSITION-LEVEL SELECTIVITY
# ══════════════════════════════════════════════════════════════════════

def compute_selectivity(
    model: BDH,
    device: str,
    best_layer: int,
    n_heads: int,
    n_neurons: int,
) -> Dict[str, Any]:
    """
    Compute neuron selectivity using POSITION-LEVEL activations from
    full sentences:

    For each concept:
    - Run all SENTENCE_CORPUS sentences
    - x_sparse at concept-word positions -> "in" activations
    - x_sparse at non-concept-word positions -> "out" activations
    - Selectivity = mean_in / (mean_in + mean_out)

    This measures whether neurons fire for concept MEANING in context,
    not just byte patterns of isolated words.  Gives ~20 concept positions
    and ~100 non-concept positions per concept for better Mann-Whitney.
    """
    from scipy.stats import mannwhitneyu

    concept_in_acts: Dict[str, Dict[int, List[np.ndarray]]] = {}
    concept_out_acts: Dict[str, Dict[int, List[np.ndarray]]] = {}

    for cid, cat in CATEGORIES.items():
        concept_words_set = {w.lower() for w in cat["words"]}
        sentences = SENTENCE_CORPUS.get(cid, [])

        in_acts: Dict[int, List[np.ndarray]] = {h: [] for h in range(n_heads)}
        out_acts: Dict[int, List[np.ndarray]] = {h: [] for h in range(n_heads)}

        for sent in sentences:
            tokens = torch.tensor(
                [list(sent.encode("utf-8"))], dtype=torch.long, device=device,
            )
            extraction_config = ExtractionConfig(
                capture_sparse_activations=True,
                capture_attention_patterns=False,
                layers_to_capture=[best_layer],
            )
            word_boundaries = _split_sentence_to_words(sent)

            with torch.no_grad():
                with model.extraction_mode(extraction_config) as buffer:
                    model(tokens)
                    x = buffer.x_sparse[best_layer][0]  # (nh, T, N)
                    T_len = x.shape[1]

                    for word, byte_start, byte_end in word_boundaries:
                        is_concept = _is_concept_word(word, concept_words_set)

                        for h in range(n_heads):
                            word_vecs = []
                            for pos in range(byte_start, min(byte_end, T_len)):
                                word_vecs.append(x[h, pos].cpu().numpy())
                            if word_vecs:
                                avg_vec = np.mean(word_vecs, axis=0)
                                if is_concept:
                                    in_acts[h].append(avg_vec)
                                else:
                                    out_acts[h].append(avg_vec)

        concept_in_acts[cid] = in_acts
        concept_out_acts[cid] = out_acts

        n_in = len(in_acts[0]) if in_acts[0] else 0
        n_out = len(out_acts[0]) if out_acts[0] else 0
        print(f"     {cid}: {n_in} concept positions, {n_out} non-concept positions")

    # Compute selectivity per neuron
    all_selectivities: List[float] = []
    per_concept_results: Dict[str, List[Dict]] = {}

    for cid in CATEGORIES.keys():
        in_acts = concept_in_acts.get(cid, {})
        out_acts = concept_out_acts.get(cid, {})
        concept_words = CATEGORIES[cid]["words"]

        monosemantic_neurons: List[Dict] = []

        for h in range(n_heads):
            in_vecs = in_acts.get(h, [])
            out_vecs = out_acts.get(h, [])
            if not in_vecs or not out_vecs:
                continue

            in_arr = np.stack(in_vecs)
            out_arr = np.stack(out_vecs)

            mean_in = in_arr.mean(axis=0)
            mean_out = out_arr.mean(axis=0)

            denom = mean_in + mean_out + 1e-10
            selectivity = mean_in / denom

            # Filter out near-dead neurons: require meaningful total activation
            # This prevents noise-driven selectivity=1.0 from near-zero neurons
            active_mask = (mean_in + mean_out) > 0.01
            active_sel = selectivity[active_mask]
            all_selectivities.extend(active_sel.tolist())

            # For the table: also require the in-concept mean to be above
            # the 10th percentile of non-zero activations (filters dead neurons)
            nz_in = mean_in[mean_in > 0]
            min_act = float(np.percentile(nz_in, 10)) if len(nz_in) > 10 else 0.01
            min_act = max(0.01, min_act)
            selective_idx = np.where(
                (selectivity > 0.6) & active_mask & (mean_in > min_act)
            )[0]

            for nidx in selective_idx:
                sel_score = float(selectivity[nidx])

                in_vals = in_arr[:, nidx]
                out_vals = out_arr[:, nidx]
                try:
                    if np.std(in_vals) > 0 or np.std(out_vals) > 0:
                        stat, pval = mannwhitneyu(in_vals, out_vals, alternative="greater")
                    else:
                        pval = 1.0
                except ValueError:
                    pval = 1.0

                per_word_vals = [round(float(v[nidx]), 5) for v in in_vecs[:len(concept_words)]]

                monosemantic_neurons.append({
                    "layer": best_layer,
                    "head": h,
                    "neuron": int(nidx),
                    "selectivity": round(sel_score, 4),
                    "mean_in": round(float(mean_in[nidx]), 5),
                    "mean_out": round(float(mean_out[nidx]), 5),
                    "p_value": round(float(pval), 8),
                    "per_word": per_word_vals,
                })

        monosemantic_neurons.sort(key=lambda n: n["selectivity"], reverse=True)
        per_concept_results[cid] = monosemantic_neurons[:30]

    n_bins = 20
    hist_counts, bin_edges = np.histogram(all_selectivities, bins=n_bins, range=(0.0, 1.0))
    histogram = [
        {
            "bin_start": round(float(bin_edges[i]), 2),
            "bin_end": round(float(bin_edges[i + 1]), 2),
            "count": int(hist_counts[i]),
        }
        for i in range(n_bins)
    ]

    total_neurons = len(all_selectivities)
    selective_count = sum(1 for s in all_selectivities if s > 0.6)

    return {
        "per_concept": per_concept_results,
        "histogram": histogram,
        "total_neurons": total_neurons,
        "total_selective": selective_count,
        "mean_selectivity": round(float(np.mean(all_selectivities)), 4)
        if all_selectivities else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════
#  PHASE 5: SYNAPSE TRACKING — cross-neuron σ(i,j)
# ══════════════════════════════════════════════════════════════════════

def compute_synapse_tracking(
    model: BDH,
    device: str,
    best_layer: int,
    n_heads: int,
    n_neurons: int,
    concepts: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Discover concept-selective Hebbian synapses σ(i,j) and track them
    word-by-word through example sentences.

    Following the BDH paper (Section 6.3, Fig 12-13):
    σ(i,j) = Σ_{τ≤t} y_sparse[τ,i] · x_sparse[τ,j]

    This is the CROSS-NEURON Hebbian outer product — neuron j's input
    activation (x_sparse) drives neuron i's output activation (y_sparse).
    The paper shows specific σ(i,j) pairs that selectively strengthen
    when the model processes concept-related words.

    Discovery algorithm:
    1. Run sentences, capture x_sparse and y_sparse per layer
    2. At concept-word positions, find top-K active neurons in both
       x_sparse (input side) and y_sparse (output side)
    3. Candidate synapses = cross-product of active y × active x
    4. Score each candidate by concept-selectivity of its Δσ
    5. Pick top 5 per concept for visualization
    """
    TOP_K_DISCOVER = 30  # neurons to consider per side
    tracking_data: Dict[str, Any] = {}

    for cid, sentences in TRACKING_SENTENCES.items():
        concept_words = set()
        for cat_id, cat in CATEGORIES.items():
            if cat_id == cid:
                concept_words = {w.lower() for w in cat["words"]}
                break

        # ── Phase A: Run sentences, collect x_sparse and y_sparse ──
        sentence_raw: List[Dict] = []

        for si, sentence in enumerate(sentences):
            tokens_bytes = list(sentence.encode("utf-8"))
            tokens_tensor = torch.tensor(
                [tokens_bytes], dtype=torch.long, device=device,
            )
            word_boundaries = _split_sentence_to_words(sentence)

            extraction_config = ExtractionConfig(
                capture_sparse_activations=True,
                capture_attention_patterns=False,
            )

            with torch.no_grad():
                with model.extraction_mode(extraction_config) as buffer:
                    model(tokens_tensor)

                    raw_per_layer: Dict[int, Dict] = {}
                    for layer_idx in sorted(buffer.x_sparse.keys()):
                        x_all = buffer.x_sparse[layer_idx]
                        y_all = buffer.y_sparse.get(layer_idx)
                        if y_all is None:
                            continue

                        x_sp = x_all[0] if x_all.dim() == 4 else x_all  # (nh, T, N)
                        y_sp = y_all[0] if y_all.dim() == 4 else y_all  # (nh, T, N)

                        raw_per_layer[layer_idx] = {
                            h: {
                                "x": x_sp[h].cpu().numpy(),  # (T, N)
                                "y": y_sp[h].cpu().numpy(),   # (T, N)
                            }
                            for h in range(n_heads)
                        }

            sentence_raw.append({
                "sentence": sentence,
                "tokens_bytes": tokens_bytes,
                "word_boundaries": word_boundaries,
                "raw_per_layer": raw_per_layer,
                "si": si,
            })

        n_sentences = len(sentences)

        # ── Phase B: Discover concept-selective cross-neuron pairs ──
        # For each layer & head, find (y_neuron, x_neuron) pairs
        # that have high Δσ at concept words and low Δσ at non-concept words

        candidate_scores: List[Dict] = []

        for layer_idx in range(model.config.n_layer):
            for h in range(n_heads):
                # Collect which neurons fire at concept vs non-concept positions
                concept_y_accum = np.zeros(n_neurons)
                concept_x_accum = np.zeros(n_neurons)
                concept_count = 0
                nonconcept_y_accum = np.zeros(n_neurons)
                nonconcept_x_accum = np.zeros(n_neurons)
                nonconcept_count = 0

                for sraw in sentence_raw:
                    layer_data = sraw["raw_per_layer"].get(layer_idx, {}).get(h)
                    if layer_data is None:
                        continue
                    x_np = layer_data["x"]  # (T, N)
                    y_np = layer_data["y"]  # (T, N)
                    T = x_np.shape[0]

                    for word, byte_start, byte_end in sraw["word_boundaries"]:
                        last_byte = min(byte_end - 1, T - 1)
                        is_concept = _is_concept_word(word, concept_words)

                        if is_concept:
                            concept_x_accum += x_np[last_byte]
                            concept_y_accum += y_np[last_byte]
                            concept_count += 1
                        else:
                            nonconcept_x_accum += x_np[last_byte]
                            nonconcept_y_accum += y_np[last_byte]
                            nonconcept_count += 1

                if concept_count == 0 or nonconcept_count == 0:
                    continue

                # Mean activations at concept vs non-concept positions
                mean_cx = concept_x_accum / concept_count
                mean_cy = concept_y_accum / concept_count
                mean_nx = nonconcept_x_accum / nonconcept_count
                mean_ny = nonconcept_y_accum / nonconcept_count

                # Top-K neurons that are preferentially active at concept words
                # Score: concept_mean - nonconcept_mean (contrast)
                x_contrast = mean_cx - mean_nx
                y_contrast = mean_cy - mean_ny

                # Also require minimum absolute activation at concept positions
                x_mask = mean_cx > 0.01
                y_mask = mean_cy > 0.01

                x_scores = np.where(x_mask, x_contrast, -1e9)
                y_scores = np.where(y_mask, y_contrast, -1e9)

                top_x_idx = np.argsort(x_scores)[-TOP_K_DISCOVER:][::-1]
                top_y_idx = np.argsort(y_scores)[-TOP_K_DISCOVER:][::-1]

                # Filter to those with positive contrast
                top_x_idx = [i for i in top_x_idx if x_scores[i] > 0]
                top_y_idx = [i for i in top_y_idx if y_scores[i] > 0]

                if not top_x_idx or not top_y_idx:
                    continue

                # Score candidate (y_i, x_j) pairs by Δσ selectivity
                # For efficiency, limit to top 15 × top 15 = 225 candidates
                for y_idx in top_y_idx[:15]:
                    for x_idx in top_x_idx[:15]:
                        # Compute Δσ(y_idx, x_idx) at each word position
                        concept_deltas = []
                        nonconcept_deltas = []
                        fired_sentences = set()

                        for sraw in sentence_raw:
                            ld = sraw["raw_per_layer"].get(layer_idx, {}).get(h)
                            if ld is None:
                                continue
                            x_np = ld["x"]
                            y_np = ld["y"]
                            T = x_np.shape[0]

                            for word, bs, be in sraw["word_boundaries"]:
                                lb = min(be - 1, T - 1)
                                fb = max(bs - 1, 0)
                                is_c = _is_concept_word(word, concept_words)

                                # Δσ at this word = sum of y[t,i]*x[t,j] over word bytes
                                delta = 0.0
                                for t in range(max(bs, 0), min(be, T)):
                                    delta += float(y_np[t, y_idx] * x_np[t, x_idx])

                                if is_c:
                                    concept_deltas.append(delta)
                                    if abs(delta) > 1e-6:
                                        fired_sentences.add(sraw["si"])
                                else:
                                    nonconcept_deltas.append(delta)

                        if not concept_deltas or not nonconcept_deltas:
                            continue

                        mc = np.mean(concept_deltas)
                        mn = np.mean(nonconcept_deltas)
                        denom = abs(mc) + abs(mn) + 1e-10
                        sel = mc / denom

                        if sel > 0.52 and mc > 1e-5:
                            candidate_scores.append({
                                "layer": int(layer_idx),
                                "head": int(h),
                                "y_neuron": int(y_idx),
                                "x_neuron": int(x_idx),
                                "selectivity": float(sel),
                                "mean_concept_delta": float(mc),
                                "mean_nonconcept_delta": float(mn),
                                "consistency": len(fired_sentences),
                            })

        # Rank candidates: consistency × selectivity × delta magnitude
        min_consistency = max(2, n_sentences // 2)
        consistent = [c for c in candidate_scores if c["consistency"] >= min_consistency]
        if not consistent:
            consistent = [c for c in candidate_scores if c["consistency"] >= 1]
        if not consistent:
            consistent = candidate_scores

        consistent.sort(
            key=lambda c: c["consistency"] * c["selectivity"] * abs(c["mean_concept_delta"]),
            reverse=True,
        )
        top_synapses = consistent[:5]

        # Fallback: if no good cross-neuron pairs, use diagonal (gate) entries
        if not top_synapses:
            print(f"     {cid}: no cross-neuron pairs found, falling back to diagonal σ(i,i)")
            for sraw in sentence_raw[:1]:
                for layer_idx in range(model.config.n_layer):
                    ld = sraw["raw_per_layer"].get(layer_idx, {})
                    for h_idx in range(n_heads):
                        hd = ld.get(h_idx)
                        if hd is None:
                            continue
                        x_np = hd["x"]
                        y_np = hd["y"]
                        diag = (x_np * y_np).sum(axis=0)  # (N,)
                        top_diag = np.argsort(diag)[-5:][::-1]
                        for nidx in top_diag:
                            if diag[nidx] > 1e-4:
                                top_synapses.append({
                                    "layer": int(layer_idx),
                                    "head": int(h_idx),
                                    "y_neuron": int(nidx),
                                    "x_neuron": int(nidx),
                                    "selectivity": 0.5,
                                    "mean_concept_delta": float(diag[nidx]),
                                    "mean_nonconcept_delta": 0.0,
                                    "consistency": 1,
                                })
                        if len(top_synapses) >= 5:
                            break
                    if len(top_synapses) >= 5:
                        break
            top_synapses = top_synapses[:5]

        if top_synapses:
            s0 = top_synapses[0]
            is_cross = s0["y_neuron"] != s0["x_neuron"]
            print(f"     {cid}: best σ({s0['y_neuron']},{s0['x_neuron']}) "
                  f"L{s0['layer']}_H{s0['head']} "
                  f"sel={s0['selectivity']:.3f} "
                  f"{'cross' if is_cross else 'diag'} "
                  f"cons={s0['consistency']}/{n_sentences}")

        # Build TrackedSynapse objects
        tracked_synapses = [
            {
                "id": f"σ({s['y_neuron']},{s['x_neuron']})",
                "label": f"L{s['layer']}_H{s['head']}_y{s['y_neuron']}_x{s['x_neuron']}",
                "layer": int(s["layer"]),
                "head": int(s["head"]),
                "i": int(s["y_neuron"]),
                "j": int(s["x_neuron"]),
                "selectivity": round(s["selectivity"], 3),
            }
            for s in top_synapses
        ]

        # ── Phase C: Word-level σ timeline for tracked synapses ──
        sentence_tracks: List[Dict] = []
        sentence_activation_scores: List[Tuple[int, float]] = []

        for si, sraw in enumerate(sentence_raw):
            word_timeline: List[Dict] = []
            T = len(sraw["tokens_bytes"])

            # For each tracked synapse, compute cumulative σ(i,j) word by word
            # σ_t(i,j) = Σ_{τ=0}^{t} y[τ,i] · x[τ,j]
            cumulative_sigma: Dict[str, float] = {syn["id"]: 0.0 for syn in tracked_synapses}
            total_activation = 0.0

            for word, byte_start, byte_end in sraw["word_boundaries"]:
                is_concept = _is_concept_word(word, concept_words)

                sigma_at_word: Dict[str, float] = {}
                delta_sigma: Dict[str, float] = {}

                for syn in tracked_synapses:
                    layer_data = sraw["raw_per_layer"].get(syn["layer"], {}).get(syn["head"])
                    if layer_data is None:
                        sigma_at_word[syn["id"]] = 0.0
                        delta_sigma[syn["id"]] = 0.0
                        continue

                    x_np = layer_data["x"]  # (T_actual, N)
                    y_np = layer_data["y"]  # (T_actual, N)
                    T_actual = x_np.shape[0]
                    y_idx = syn["i"]
                    x_idx = syn["j"]

                    # Δσ = sum of y[t,i]*x[t,j] over the bytes of this word
                    ds = 0.0
                    for t in range(max(byte_start, 0), min(byte_end, T_actual)):
                        ds += float(y_np[t, y_idx] * x_np[t, x_idx])

                    cumulative_sigma[syn["id"]] += ds
                    sigma_at_word[syn["id"]] = round(cumulative_sigma[syn["id"]], 6)
                    delta_sigma[syn["id"]] = round(ds, 6)

                    if is_concept:
                        total_activation += abs(ds)

                word_timeline.append({
                    "word": word,
                    "byte_start": byte_start,
                    "byte_end": byte_end,
                    "is_concept": is_concept,
                    "sigma": sigma_at_word,
                    "delta_sigma": delta_sigma,
                })

            sentence_tracks.append({
                "sentence": sraw["sentence"],
                "n_bytes": len(sraw["tokens_bytes"]),
                "words": word_timeline,
            })
            sentence_activation_scores.append((si, total_activation))

        # Sort sentences by total concept activation (best evidence first)
        sentence_activation_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_tracks = [sentence_tracks[idx] for idx, _ in sentence_activation_scores]

        tracking_data[cid] = {
            "synapses": tracked_synapses,
            "sentences": sorted_tracks,
        }

    return tracking_data


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute monosemanticity data for BDH frontend"
    )
    parser.add_argument("--model", default=str(ROOT / "checkpoints" / "french" / "french_best.pt"))
    parser.add_argument("--output", default=str(ROOT / "frontend" / "public" / "monosemanticity" / "precomputed.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("\U0001f9e0 BDH Monosemanticity Pre-computation V2")
    print("   Sentence-level extraction \u2022 Position-level selectivity")
    print("=" * 60)

    print(f"\n\U0001f4c2 Loading model from {args.model}")
    model = load_model(args.model, args.device)
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    n_neurons = model.config.n_neurons
    print(f"   Config: {n_layers}L \u00d7 {n_heads}H \u00d7 {n_neurons}N")

    # ── Phase 1: Sentence-level fingerprints ──
    print("\n" + "=" * 60)
    print("Phase 1: Sentence-level concept fingerprints")
    print("=" * 60)
    concepts: Dict[str, Any] = {}
    for cat_id, cat in CATEGORIES.items():
        sentences = SENTENCE_CORPUS.get(cat_id, [])
        print(f"\n   \u25b6 {cat['name']}: {cat['words']} ({len(sentences)} sentences)")
        result = extract_fingerprint(
            model, cat["words"], sentences, cat["name"],
            args.device, n_layers, n_heads, n_neurons,
        )
        concepts[cat_id] = result
        for layer_str, matrix in result["similarity"].items():
            n = len(matrix)
            avg = sum(matrix[i][j] for i in range(n) for j in range(n) if i != j) / max(n * (n - 1), 1)
            print(f"     L{layer_str} avg cosine: {avg:.4f}")

    # ── Phase 2: Best layer ──
    best_layer = compute_best_layer(concepts, n_layers)
    print(f"\n\U0001f3c6 Best layer: L{best_layer}")

    # ── Phase 3: Cross-concept ──
    print("\n" + "=" * 60)
    print("Phase 3: Cross-concept distinctness")
    print("=" * 60)
    cross_concept = compute_cross_concept(concepts, n_layers, n_heads)
    for cc in cross_concept:
        avg_d = np.mean(cc["distinctness_per_layer"])
        print(f"   {cc['primary']} vs {cc['secondary']}: avg distinctness = {avg_d:.4f}")

    # ── Phase 4: Position-level selectivity ──
    print("\n" + "=" * 60)
    print("Phase 4: Position-level neuron selectivity")
    print("=" * 60)
    selectivity_data = compute_selectivity(model, args.device, best_layer, n_heads, n_neurons)

    for cid, neurons in selectivity_data["per_concept"].items():
        concepts[cid]["monosemantic_neurons"] = neurons
        sig_count = sum(1 for n in neurons if n["p_value"] < 0.05)
        print(f"   {cid}: {len(neurons)} selective, {sig_count} significant (p<0.05)")

    print(f"\n   \U0001f4ca Global: {selectivity_data['total_neurons']} neurons, "
          f"{selectivity_data['total_selective']} selective (>0.6), "
          f"mean = {selectivity_data['mean_selectivity']:.4f}")

    # ── Phase 5: Synapse tracking ──
    print("\n" + "=" * 60)
    print("Phase 5: Synapse tracking (\u03c3 word-by-word)")
    print("=" * 60)
    synapse_tracking = compute_synapse_tracking(
        model, args.device, best_layer, n_heads, n_neurons, concepts,
    )
    for cid, track in synapse_tracking.items():
        n_syn = len(track["synapses"])
        n_sent = len(track["sentences"])
        print(f"   {cid}: {n_syn} synapses \u00d7 {n_sent} sentences")

    # ── Phase 6: Write JSON ──
    for cid in concepts:
        concepts[cid].pop("raw_x", None)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_info": {"n_layers": n_layers, "n_heads": n_heads, "n_neurons": n_neurons},
        "best_layer": best_layer,
        "concepts": concepts,
        "cross_concept": cross_concept,
        "selectivity": {
            "histogram": selectivity_data["histogram"],
            "total_neurons": selectivity_data["total_neurons"],
            "total_selective": selectivity_data["total_selective"],
            "mean_selectivity": selectivity_data["mean_selectivity"],
        },
        "synapse_tracking": synapse_tracking,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n\U0001f4be Wrote {output_path} ({size_mb:.2f} MB)")
    print("\u2705 Done!")


if __name__ == "__main__":
    main()