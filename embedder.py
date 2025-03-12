# embedder.py

import os
from typing import List

import openai
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

class TextEmbedder:
    """
    A class for generating embeddings using OpenAI's API or a local model (SentenceTransformers).
    """

    def __init__(self, model_name="text-embedding-ada-002", use_openai=True):
        """
        Initialize the embedder with a selected model.

        :param model_name: str - Model name for embeddings. Default is OpenAI's ada-002.
        :param use_openai: bool - Whether to use OpenAI API (True) or local SentenceTransformers (False).
        """
        self.use_openai = use_openai
        self.model_name = model_name

        if use_openai:
            
            self.api_key = os.getenv("OPENAI_API_KEY")  # Load API key from environment
            if not self.api_key:
                raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY in the environment.")
            self.client = OpenAI(api_key=self.api_key)  # Define self.client
        else:
            # Load local embedding model
            self.local_model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text inputs.

        :param texts: List[str] - List of text chunks to embed.

        :return: List[List[float]] - List of embedding vectors.
        """
        if self.use_openai:
            return self._embed_with_openai(texts)
        else:
            return self._embed_with_local_model(texts)

    def _embed_with_openai(self, texts: List[str]) -> List[List[float]]:
        """
        Uses OpenAI's API to generate embeddings.

        :param texts: List[str] - List of text chunks.

        :return: List[List[float]] - List of embedding vectors.
        """
        texts =""" 

Wikipedia L'enciclopedia libera

    Fai una donazione
    registrati
    entra

Banner logo

È aperta la selezione per le proposte per il programma di Wikimania 2025!
Clicca qui per candidare una proposta
Puoi candidare una proposta fino alle 23:59 del 31 marzo 2025
[ Dai una mano con le traduzioni! ]
Indice
Inizio
Storia

Descrizione

        Implementazione sui sistemi di calcolo
        Problemi
        Algoritmi
    Applicazione ed esempi di utilizzo
    Soluzioni per l'HPC
    Note
    Bibliografia
    Voci correlate
    Collegamenti esterni

High performance computing

    Voce
    Discussione

    Leggi
    Modifica
    Modifica wikitesto
    Cronologia

Strumenti

Aspetto
Testo

    Piccolo
    Standard
    Grande

Larghezza

    Standard
    Largo

Colore (beta)

    Automatico
    Chiaro
    Scuro

Con high performance computing (HPC) (in italiano calcolo ad elevate prestazioni), in informatica, ci si riferisce alle tecnologie utilizzate da computer cluster per creare dei sistemi di elaborazione in grado di fornire delle prestazioni molto elevate nell'ordine dei PetaFLOPS, ricorrendo tipicamente al calcolo parallelo.

L'espressione è molto utilizzata essenzialmente per sistemi di elaborazioni utilizzati in campo scientifico.

Gli attuali sistemi di calcolo più diffusi, che sfruttano le tecnologie HPC, sono installazioni che richiedono rilevanti investimenti e la cui gestione richiede l'utilizzo di personale specializzato di alto livello. L'intrinseca complessità e rapida evoluzione tecnologica di questi strumenti richiede, inoltre, che tale personale interagisca profondamente con gli utenti finali (gli esperti dei vari settori scientifici nei quali questi sistemi vengono utilizzati), per consentire loro un utilizzo efficiente degli strumenti [1].
Storia

Il 1990 vide la nascita del primo modello standard di programmazione parallela per HPC. All'inizio del decennio, i sistemi di supercalcolo vettoriale come quelli commercializzati dalla Cray Research, Fujitsu e NEC, erano ampiamente utilizzati nell'esecuzione di applicazioni su larga scala. Venivano combinati insieme da due a quattro processori vettoriali che formavano sistemi particolarmente potenti con una singola memoria condivisa. I multiprocessori simmetrici (SMP), erano costituiti da un piccolo numero di processori RISC che condividevano la memoria, ma sorsero dei problemi quando fu chiaro che sarebbe stato difficile estendere questa tecnologia ad un grande numero di CPU. Nacquero così le nuove piattaforme parallele a memoria distribuita (DMP) prodotte da compagnie come Intel, Meiko e nCube: i computer SIMD (Single Instruction Multiple Data) che potevano eseguire una singola istruzione su un insieme di dati simultaneamente, ad esempio sugli elementi di un array. Questi nuovi sistemi, benché costosi da acquistare e gestire, potevano essere costruiti in formati differenti così da poter creare macchine dal costo differenziato in base alle esigenze e i budget delle aziende clienti.

A poco a poco le aziende iniziarono a produrre delle DMP proprie costituite da una serie di singoli computer collegati ad una rete ad alta velocità con un sistema che garantisse un rapido trasporto di dati tra le diverse memorie. Allo stesso modo le workstation collegate a delle comuni LAN iniziarono ad essere utilizzate insieme per lo svolgimento di un unico lavoro, fornendo così dei sistemi paralleli a basso costo. Tali sistemi divennero noti come “Clusters of workstation” (COW). Sebbene le reti Ethernet usati dalle COW fossero lente in confronto a quelle delle vere DMP, queste risultavano però molto più economiche da costruire. Nella seconda metà degli anni novanta, la maggior parte dei produttori statunitensi cominciarono a produrre SMP. Contrariamente a quelli costruiti negli anni ottanta, questi ultimi relativamente poco costosi, erano destinati ad un più ampio impiego, come i computer desktop.

Nell'ultimo decennio del XX secolo l'hardware si è sviluppato notevolmente. La frequenza di clock dei processori è salita da alcune decine, fino a migliaia di Megahertz e le dimensioni delle memorie principali e di massa sono cresciute di diversi ordini di grandezza. Alcune applicazioni, che fino ad allora avrebbero richiesto un hardware per HPC, poterono essere eseguite con un solo SMP. Per la prima volta i computer paralleli divennero accessibili ad un'ampia fascia di utenti. Infine tra le altre alternative esistono delle piattaforme HPC che eseguono il codice su una CPU accedendo direttamente ai dati memorizzati su un altro sistema SMP, utilizzando ad esempio un indirizzamento globale ed un sistema per il supporto della rete di collegamento tra i vari SMP. Tali sistemi sono i cosiddetti ccNUMA (cache coherent NonUniform Memory Access). Questi ultimi possono essere visti come delle grandi SMP virtuali, anche se il codice viene eseguito più lentamente se i dati su cui si opera sono memorizzati su un altro SMP (da qui il costo d'accesso non uniforme).

Oggi i supercomputer vettoriali continuano a fornire i più alti livelli di prestazioni per alcune applicazioni, ma restano comunque costosi. Le SMP sono largamente commercializzate, con un numero di processori che va da due a oltre cento, come il Sun Modular Datacenter della Sun. La maggior parte dei commercianti di hardware, combina adesso le tecnologie DMP e SMP per soddisfare la richiesta di soluzioni per il calcolo su larga scala. I COW sono diventati comuni e i cluster, composti da migliaia di processori, vengono utilizzati sia nei laboratori così come nell'industria (ad esempio per l'elaborazione sismica). Il costo delle reti è nel frattempo diminuito mentre le prestazioni fornite sono aumentate rapidamente, così che il problema associato alla lentezza delle reti COW si è di molto ridotto [2]. La tecnologia InfiniBand è la più utilizzata per interconnettere i sistemi HPC anche se la tecnologia 10 gigabit Ethernet con l'aggiunta delle tecnologie lossless ethernet tendenzialmente potrà essere la soluzione di riferimento.[3] A rendere più complicato il panorama del settore HPC c'è la tendenza alla creazione di sistemi ibridi, non più trainati solo dall'aumento del numero di cpu presenti, ma anche da altri coprocessori (come schede GPU o processori FPGA) che, andandosi ad affiancare alle cpu, hanno aperto nuove potenzialità di calcolo.
Descrizione

I sistemi di calcolo più diffusi, che sfruttano le tecnologie HPC, sono installazioni che richiedono rilevanti investimenti e la cui gestione richiede l'utilizzo di personale specializzato di alto livello. L'intrinseca complessità e rapida evoluzione tecnologica di questi strumenti richiede, inoltre, che tale personale interagisca profondamente con gli utenti finali (gli esperti dei vari settori scientifici nei quali questi sistemi vengono utilizzati), per consentire loro un utilizzo efficiente degli strumenti [1].

È importante evidenziare la sottile differenza tra High Performance Computing (HPC) e "supercomputer". HPC è un termine, talvolta usato come sinonimo di supercomputer, che è sorto dopo il termine "supercomputing" (supercalcolo). In altri contesti, "supercomputer" è usato per riferirsi ad un sottoinsieme di "computer ad alte prestazioni", mentre il termine "supercomputing" si riferisce ad una parte del "calcolo ad alte prestazioni" (HPC). La possibile confusione circa l'uso di questi termini è evidente.
Implementazione sui sistemi di calcolo

Poiché i sistemi di calcolo, ogni giorno, sono sempre più sofisticati e veloci, gli sviluppatori di applicazioni per HPC devono spesso lavorare insieme ad ingegneri e progettisti per identificare e correggere i vari bug e le instabilità che insorgono. Essi devono adeguarsi continuamente a scrivere codice per nuovi tipi di architetture e spesso sono i primi che utilizzano i nuovi linguaggi di programmazione, le librerie, i compilatori, insieme ai più recenti strumenti per lo sviluppo di applicazioni. Tuttavia, la continua riprogrammazione si rivela inefficiente perché impiega troppa manodopera e gli esperti in applicazioni HPC non sono molti, dunque sono essenziali dei modelli standard di programmazione ad alto livello per ridurre lo sforzo umano nella riprogrammazione delle nuove piattaforme.

Purtroppo fornire standard ad alto livello per le applicazioni HPC non è un'impresa facile poiché nascono ogni giorno nuove architetture per il calcolo ad alte prestazioni, ognuna con specifiche caratteristiche diverse dalle altre e che devono essere sfruttate in maniera adeguata per raggiungere l'alto livello di prestazione richiesto.

I programmatori HPC sono riluttanti a sacrificare le prestazioni in cambio di una maggiore facilità di programmazione, dunque è necessario un modello standard che permetta di sfruttare al massimo tutti i tipi di piattaforma. Vi sono poi ulteriori vincoli: il modello di programmazione deve essere di facile comprensione per semplificare l'identificazione dei bug e la loro correzione. Deve essere scalabile, per far fronte all'aumento di complessità del problema e quello di potenza della macchina. Infine, poiché quello dell'HPC è una mercato relativamente piccolo, i produttori non sarebbero in grado di far fronte ad un'elevata quantità di modelli di programmazione diversi e di conseguenza, lo standard, deve essere compatibile con la maggior parte dei codici e piattaforme per HPC [2].
Problemi

I programmatori abituati a ragionare in modo sequenziale hanno dovuto acquisire una nuova mentalità e, anche se parte della conversione da software applicativo sequenziale a parallelo poteva essere eseguita in maniera automatica, restava pur sempre da svolgere un'ulteriore attività di “parallelizzazione” manuale che poteva richiedere la ristrutturazione di algoritmi che erano stati pensati in modo essenzialmente seriale. Inoltre durante l'evoluzione dell'High Performance Computing è emerso un fattore tecnologico che, poco alla volta, è diventato uno dei maggiori vincoli alla crescita dell'effettiva capacità di calcolo dei supercomputer. Questo vincolo è rappresentato dal progressivo sbilanciamento tra la tecnologia dei processori e quella della memoria dei supercomputer ovvero la velocità di esecuzione delle istruzioni è aumentata molto più rapidamente del tempo di accesso alla memoria centrale.[4]
Algoritmi

È necessario considerare l'importanza degli algoritmi. Quest'ultimo è un aspetto spesso sottovalutato. Infatti, per utilizzare al meglio un computer che esegua contemporaneamente su molteplici complessi circuitali svariati gruppi di istruzioni occorre risolvere due problemi. È necessario, innanzitutto, sviluppare in software un algoritmo che si presti a essere suddiviso in più parti, ossia in diverse sequenze di istruzioni, da eseguire in parallelo. In secondo luogo, occorre disporre di un linguaggio ed un compilatore, che sappia ottimizzare nelle giuste sequenze da distribuire in parallelo le istruzioni scritte dal programmatore[4]. Tra questi linguaggi abbiamo ad esempio l'Occam.

La progettazione di un algoritmo efficiente è, spesso, più efficace di un hardware sofisticato. Purtroppo, non è facile trovare la soluzione giusta. Con la disponibilità del calcolo parallelo, il ricercatore che debba sviluppare una nuova applicazione potrebbe essere invogliato a cercare fin dall'inizio algoritmi intrinsecamente parallelizzabili. Più difficile è, evidentemente, il compito di chi debba riadattare software applicativo già esistente alle nuove possibilità tecnologiche.

Bisogna inoltre considerare i limiti dell'elaborazione parallela. A titolo d'esempio, si immagini di avere 52 carte di uno stesso mazzo distribuite in modo casuale e di volerle mettere in ordine. Se ci fossero quattro giocatori a volersi dividere le attività secondo i colori (cuori, quadri, fiori, picche) si farebbe certamente prima che se si dovesse svolgere la stessa attività da soli. Ma fossero 52 i giocatori, tutti intorno allo stesso tavolo, a voler ordinare il mazzo si farebbe probabilmente solo una gran confusione. Questo banale esempio può far comprendere come esista un limite al di là del quale non conviene spingere sul parallelismo degli agenti ma, piuttosto, sulla velocità del singolo [4].

L'introduzione di infrastrutture eterogenee di calcolo, come per esempio tecnologie gpgpu, ha alzato ancor di più il livello di complessità nella progettazione e creazione di algoritmi che siano in grado di sfruttarle adeguatamente.
Applicazione ed esempi di utilizzo

Sebbene i modelli matematici applicati all'astrazione e modellizzazione di sistemi e di fenomeni siano stati in alcuni casi elaborati da molti decenni, solo recentemente, grazie all'avvento di piattaforme di calcolo ad alte prestazioni, hanno avuto modo di mostrare il loro enorme potere esplicativo e predittivo in molti ambiti scientifici. Le moderne tecnologie informatiche hanno, infatti, consentito un enorme sviluppo delle tecniche di modellistica numerica fornendo uno straordinario contributo negli ultimi decenni, sia all'avanzamento della conoscenza, che alla realizzazione di prodotti e processi tecnologicamente avanzati.

Queste hanno reso possibile progettare, studiare, riprodurre e visualizzare complessi fenomeni naturali e sistemi ingegneristici con un'accuratezza fino a pochi anni fa impensabile. Si è sviluppata nel tempo una nuova categoria di specialisti in modellistica computazionale; questi sono in genere esperti nei vari domini applicativi con forti competenze nell'informatica avanzata che rendono possibile l'utilizzo di questi strumenti all'interno dei gruppi di ricerca delle varie aree applicative [1].

Il calcolo ad alte prestazioni viene utilizzato in svariati settori, e per gli scopi più disparati tra questi abbiamo ad esempio:

    Lo studio del clima globale in Climatologia;
    Le equazioni fluidodinamiche della Fisica;
    Lo studio della materia a livello atomico (Equazione di Schrödinger) nel settore della Chimica;
    La ricerca di metodi di conservazione di antichi testi e scritture per l'Archeologia;
    Lo sviluppo di nuovi farmaci in Farmacologia;
    Lo studio delle proteine in Medicina, questo molto importante per una futura cura contro la malattie degenerative;
    L'analisi di dati genomici

Questi sono comunque solo alcune delle possibili applicazioni dell'HPC[1][2][4].
Soluzioni per l'HPC

    UniClust HPC Suite [collegamento interrotto], su unicluster.fis.unical.it.
    Sun Modular Datacenter, su sun.com.
    Windows HPC, su microsoft.com.

Note

Vincenzo Artale, Massimo Celino, Calcolo numerico ad alte prestazioni (PDF), su afs.enea.it, 2008.
(EN) Laurence Tianruo Yang, Guo Minyi, The Challenge of Providing a High-level Programming Model for High performance computing, in High-performance computing: paradigm and infrastructure, Wiley Interscienze, 2006, ISBN 978-0-471-65471-1. URL consultato il 23 febbraio 2010.
^ (EN) Silvano Gai, I/O Consolidation in the Data Center, in Data Center Networks and Fibre Channel over Ethernet (FCoE), California (USA), Lulu.com, aprile 2008, ISBN 978-1-4357-1424-3. URL consultato il 18 giugno 2009 (archiviato dall'url originale il 3 marzo 2009).

    Ernesto Hoffman, Evoluzione e prospettive dell'High Performance Computing (PDF), su mondodigitale.net, 2003. URL consultato il 1º marzo 2010 (archiviato dall'url originale il 6 maggio 2006).

Bibliografia

    Laurence Tianruo Yang, Minyi Guo, High-performance computing: paradigm and infrastructure, Wiley Interscienze, 2006. ISBN 978-0-471-65471-1.
    Evoluzione e prospettive dell'High performance computing, Ernesto Hoffman.
    Calcolo numerico ad alte prestazioni, Vincenzo Artale, Massimo Celino.

Voci correlate

    Grid computing
    Cloud computing
    Supercomputer
    TOP500
    GPGPU
    High Performance Fuzzy Computing

Collegamenti esterni

    (EN) Opere riguardanti High performance computing, su Open Library, Internet Archive. Modifica su Wikidata
    (EN) International Conference On High Performance Computing, su hipc.org.
    (EN) Maui High Performance Computing Center, su mhpcc.edu. URL consultato il 1º marzo 2010 (archiviato dall'url originale il 27 marzo 2010).
    Istituto di calcolo e reti ad alte prestazioni, su icar.cnr.it. URL consultato il 1º marzo 2010 (archiviato dall'url originale l'8 marzo 2010).
    HPC su GPU Computing, su hwupgrade.it.
    (EN) HPCwire, su HPCwire.com.
    (EN) Top 500 supercomputers, su top500.org.
    (EN) Rocks Clusters Open-Source High Performance Linux Clusters
    (EN) Infiscale Abstractual and Perceus Open-Source Extreme Scale HPC Clusters and Clouds
    Linux ParallelKnoppix, su parallelknoppix.info. URL consultato il 20 aprile 2011 (archiviato dall'url originale l'8 maggio 2011).
    Windows HPC Server 2008, su microsoft.com.

Controllo di autorità	LCCN (EN) sh95008935 · GND (DE) 4532701-4 · J9U (EN, HE) 987007563508405171
  Portale Informatica
  Portale Telematica
Categoria:

    Supercomputer

[altre]

    Questa pagina è stata modificata per l'ultima volta il 5 ott 2024 alle 01:54.
    Il testo è disponibile secondo la licenza Creative Commons Attribuzione-Condividi allo stesso modo; possono applicarsi condizioni ulteriori. Vedi le condizioni d'uso per i dettagli.

    Informativa sulla privacy
    Informazioni su Wikipedia
    Avvertenze
    Codice di condotta
    Sviluppatori
    Statistiche
    Dichiarazione sui cookie
    Versione mobile

    Wikimedia Foundation
    Powered by MediaWiki

High performance computing
Aggiungi argomento


"""

        try:
            response = self.client.embeddings.create(input=texts,
            model=self.model_name)
            return [item.embedding for item in response.data]

        except openai.OpenAIError as e:
            print(f"OpenAI API Error: {e}")
            return []

    def _embed_with_local_model(self, texts: List[str]) -> List[List[float]]:
        """
        Uses a local SentenceTransformers model to generate embeddings.

        :param texts: List[str] - List of text chunks.

        :return: List[List[float]] - List of embedding vectors.
        """
        return self.local_model.encode(texts, convert_to_numpy=True).tolist()


# TODO
# Using a Local Model (sentence-transformers)
# If you want to avoid OpenAI API costs and use a local embedding model, initialize it with:


# if __name__ == "__main__":
#     embedder = TextEmbedder(model_name="all-MiniLM-L6-v2", use_openai=False)  # Local model
#     texts = ["HPC clusters enable parallel computing.", "Slurm is a job scheduler."]

#     embeddings = embedder.embed_texts(texts)
#     print(embeddings[:2])
# Recommended Local Models:

# "all-MiniLM-L6-v2" → Fast & small (~80MB)
# "all-mpnet-base-v2" → More accurate but slower (~400MB)


if __name__ == "__main__":
    embedder = TextEmbedder(use_openai=True)  # Uses OpenAI API
    texts = ["HPC clusters enable parallel computing.", "Slurm is a job scheduler."]

    embeddings = embedder.embed_texts(texts)
    print(embeddings[:2])  # Print first two embeddings
