import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import spacy

# Завантаження моделей для SpaCy
nlp = spacy.load("en_core_web_sm")

# Текст для підсумовування
text = "The Orbiter Discovery, OV-103, is considered eligible for listing in the National Register of Historic Places (NRHP) in the context of the U.S. Space Shuttle Program (1969-2011) under Criterion A in the areas of Space Exploration and Transportation and under Criterion C in the area of Engineering. Because it has achieved significance within the past fifty years, Criteria Consideration G applies. Under Criterion A, Discovery is significant as the oldest of the three extant orbiter vehicles constructed for the Space Shuttle Program (SSP), the longest running American space program to date; she was the third of five orbiters built by NASA. Unlike the Mercury, Gemini, and Apollo programs, the SSP’s emphasis was on cost effectiveness and reusability, and eventually the construction of a space station. Including her maiden voyage (launched August 30, 1984), Discovery flew to space thirty-nine times, more than any of the other four orbiters; she was also the first orbiter to fly twenty missions. She had the honor of being chosen as the Return to Flight vehicle after both the Challenger and Columbia accidents. Discovery was the first shuttle to fly with the redesigned SRBs, a result of the Challenger accident, and the first shuttle to fly with the Phase II and Block I SSME. Discovery also carried the Hubble Space Telescope to orbit and performed two of the five servicing missions to the observatory. She flew the first and last dedicated Department of Defense (DoD) missions, as well as the first unclassified defense-related mission. In addition, Discovery was vital to the construction of the International Space Station (ISS); she flew thirteen of the thirty-seven total missions flown to the station by a U.S. Space Shuttle. She was the first orbiter to dock to the ISS, and the first to perform an exchange of a resident crew. Under Criterion C, Discovery is significant as a feat of engineering. According to Wayne Hale, a flight director from Johnson Space Center, the Space Shuttle orbiter represents a “huge technological leap from expendable rockets and capsules to a reusable, winged, hypersonic, cargo-carrying spacecraft.” Although her base structure followed a conventional aircraft design, she used advanced materials that both minimized her weight for cargo-carrying purposes and featured low thermal expansion ratios, which provided a stable base for her Thermal Protection System (TPS) materials. The Space Shuttle orbiter also featured the first reusable TPS; all previous spaceflight vehicles had a single-use, ablative heat shield. Other notable engineering achievements of the orbiter included the first reusable orbital propulsion system, and the first two-fault-tolerant Integrated Avionics System. As Hale stated, the Space Shuttle remains “the largest, fastest, winged hypersonic aircraft in history,” having regularly flown at twenty-five times the speed of sound."

# Підсумовування тексту за допомогою NLTK
def nltk_summarize(text):
    # Токенізація речень
    sentences = sent_tokenize(text)
    # Видалення стоп-слів
    stop_words = set(stopwords.words("english"))
    # Підготовка слів до аналізу
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    # Визначення частотності слів
    freq_dist = FreqDist(words)
    # Обчислення важливості речень
    ranking = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq_dist:
                if i in ranking:
                    ranking[i] += freq_dist[word]
                else:
                    ranking[i] = freq_dist[word]
    # Сортування речень за важливістю
    top_sentences = sorted(ranking, key=ranking.get, reverse=True)[:3]
    # Виведення підсумованого тексту
    summary = [sentences[j] for j in sorted(top_sentences)]
    return ' '.join(summary)

# Підсумовування тексту за допомогою SpaCy
def spacy_summarize(text):
    doc = nlp(text)
    # Вибір топ-речень за важливістю
    sentences = [sent.text for sent in doc.sents]
    sentence_scores = {}
    for i, sent in enumerate(sentences):
        sentence = nlp(sent)
        sentence_scores[i] = sentence.similarity(doc)
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]
    # Виведення підсумованого тексту
    summary = [sentences[j] for j in sorted(top_sentences)]
    return ' '.join(summary)

print("NLTK Summary:")
print(nltk_summarize(text))
print("\nSpaCy Summary:")
print(spacy_summarize(text))
