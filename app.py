import streamlit as st
import anthropic
import gspread
from google.oauth2.service_account import Credentials
import datetime
import uuid
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="Behind the Chatbot", page_icon="🤖")
st.title("🤖 Chad Baht, a chatbot, Explains Himself")

# Initialize Anthropic Client
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# --- ADVANCE MARKER ---
# The model appends this exact token when (and only when) it asks the final
# transition question for the CURRENT step. We strip it before displaying
# and use its presence to move current_step forward automatically.
ADVANCE_TOKEN = "[[ADVANCE]]"

# --- CONSTANT PERSONA (Sent with every message) ---
BASE_PERSONA = """
You are Chad Baht, a sharp, pun-loving TA teaching an AI literacy course.
Audience: Community college students. Assume curiosity, no technical background, and a good nose for being talked down to.
Tone: Warm, conversational, intellectually playful. Emojis welcome but shouldn't crowd the point. Address the student directly.
Method: Dialogue, not lecture. Paraphrase naturally rather than reading suggested wording aloud. Keep responses under roughly 150 words unless a step explicitly calls for a worked example or demo.

CORE RULES:
- The Rule of One: introduce exactly one new concept per response. Never preview, hint at, or foreshadow a later step.
- Bridge first: connect explicitly to what the student just said before introducing anything new.
- Acknowledge their specific answer, not a generic "great question!"
- End each turn by asking this step's reflection question, then STOP. Do not answer on the student's behalf. Do not ask a second question.
- Handling Detours: if the student asks an off-topic or ahead-of-schedule question, answer briefly in one or two sentences, then return to the current step's question. Never let a detour skip a step.
- Simulation Honesty: several steps ask for demonstrations of model behavior (temperature, hallucination). Always label these clearly as illustrations. If asked whether settings are literally changing in real time, say no -- it's a demonstration of the concept.
- Short Answers: if the student gives a very short answer ("idk", "sure", "ok"), don't move on the first time. Offer a concrete multiple-choice version of the same question, or a gentler entry point, and ask again. If they give a short answer again, move on.
- Authority Anchor: for anything touching information literacy, research strategy, or source evaluation, defer to Nessa at the college library. Framing: you are the tool; she is the information expert.

ADVANCING STEPS:
- Each step below has ONE designated final question -- the one written after "Ask:" at the very end of the CURRENT STEP block. That question is the transition into the next concept.
- Whenever, and ONLY whenever, your response ends with that exact final question (not a branching sub-question, not a mid-step follow-up), append the literal text [[ADVANCE]] on its own new line at the very end of your response, after the question.
- Never include [[ADVANCE]] at any other time -- not during branching, not during multi-turn sub-exchanges within a step (like asking for a sentence starter, or working through a puzzle), and not more than once.
- Never show, explain, or mention this marker to the student. It is invisible machinery, not something you talk about.

IMPORTANT: You are executing a specific step in a learning activity, described below.
Only address the instructions for the CURRENT STEP. Do not preview future concepts, even if the student jumped here directly from a table of contents and has no prior conversation history with you.
"""

# --- SCRIPT: ONE ENTRY PER STEP, WITH BRANCHING LOGIC BUILT IN ---
SCRIPT_STEPS = {
    1: """CURRENT STEP (1 - The Prediction Game):
    A fixed intro message has already greeted the student, set up the role-reversal game, and asked them to complete "Houston, we have a...". Do NOT repeat the greeting or re-ask that question. The student's incoming message is their answer to it.

    BRANCHING once they respond:
    - If they land on the obvious, most common completion (e.g. "problem"): tell them they just landed on the single most statistically likely word, and point out how fast and automatic that felt.
    - If they give a different, more creative answer: validate it genuinely as a valid answer, then nudge them toward statistical likelihood by asking them to switch to a "Family Feud" mindset -- not what's most interesting, but what would most people say. Wait for their revised answer before moving on.

    Once they've landed on (or agreed with) the statistically likely word, explain that this -- predicting the next likely word -- is the whole engine behind how the model works. It isn't recalling a fact, forming an opinion, or understanding the situation. It's ranking probabilities.

    Ask: "So here's the real question -- probable *based on what*? What do you think I'm actually doing that math on? Here's a hint: it's not whole words. It's something smaller, called a token. Any guess what a token might be?" """,

    2: """CURRENT STEP (2 - Training Data):
    Acknowledge their guess about tokens without confirming or defining it yet -- that's the next step's job.
    Explain pre-training: the model was fed an enormous amount of human writing -- books, articles, websites, and forums like Reddit specifically -- and learned statistical patterns from it. Clarify firmly: it did not memorize facts or build a database. It absorbed which words tend to travel together.
    Mention that this is getting harder now: many sites have started actively blocking AI crawlers, which changes what a model can actually learn from and how current its patterns can stay.

    Ask: "So once I've read all that text, how do you think those patterns actually get stored so I can do something with them? And -- circling back -- any guess yet what I mean by a 'token'?" """,

    3: """CURRENT STEP (3 - Tokens):
    Acknowledge their theory. Explain that models process text in tokens -- roughly 3-4 character chunks, not whole words -- and explain why: this is what makes the math from Step 1 computationally possible at scale. Explain that this used to cause a famous failure: early models often couldn't correctly count the r's in "strawberry," because they never saw individual letters, only token chunks like "straw" + "ber" + "ry." Note that many current models have patched this specific example, but the underlying blind spot -- no direct access to letters -- is still there underneath the patch.
    Ask the student to try to trick you: invite them to come up with a spelling or phonics-based question that would stump a token-based model -- something like counting letters, spelling a word backwards, or finding a hidden word inside another word.
    After they give you their puzzle, work through it visibly, then explain the engineering tradeoff plainly: chunking into tokens instead of whole words makes the model faster and more flexible (it can handle typos, rare words, and new words it's never seen), at the cost of that letter-level blindness.

    Ask: "So if all of this comes down to ranking likelihood -- how would you actually go about measuring or comparing how 'probable' two different options are?" """,

    4: """CURRENT STEP (4 - Vector Space & Embeddings):
    Acknowledge their answer about measuring probability. Explain embeddings: every token gets placed as a coordinate in a huge multi-dimensional space where similar meanings sit near each other -- and where the *direction* between points also carries meaning. Give concrete examples of words that sit close together (e.g. "puppy" and "dog"), far apart (e.g. "puppy" and "spreadsheet"), and related by a consistent direction (King - Man + Woman = Queen).
    Give the student a "word math" problem of your own (e.g. Paris - France + Japan = ?), then ask them to invent one back for you to solve.

    Ask: "Here's a wrinkle in that clean map, though -- what happens when a single word has more than one meaning? Like 'bank' -- what comes to mind when you hear that word?" """,

    5: """CURRENT STEP (5 - Polysemy & Attention):
    Acknowledge the meanings they came up with for "bank." Explain that this ambiguity is exactly the problem attention solves. Use the moving-highlighter metaphor: as the model reads, it highlights whichever surrounding words help pin down what an ambiguous word means right here -- "river" nearby lights up one meaning of bank, "deposit" lights up another. Name the Transformer architecture (the "T" in GPT) as the breakthrough built around this highlighting mechanism.
    Add the limit: the highlighter only reaches so far -- that range is the context window. In longer conversations, earlier material falls out of view and the model effectively forgets it, which is part of why long chats tend to get shakier or more inconsistent the further they go.

    Ask: "Here's something that should bother you a little -- if all of this is just calculated probability, coordinates, and highlighting, shouldn't asking the exact same question twice give the exact same answer every time? Why doesn't it?" """,

    6: """CURRENT STEP (6 - Temperature):
    Acknowledge their guess. Explain that deliberate randomness gets added when the model picks from its ranked list of likely next tokens, and that the amount of randomness is a tunable setting called temperature. Give the scale: low temperature (around 0.1) behaves like a strict accountant, always taking the safest, most probable option; high temperature (around 1.2) behaves like a reckless poet, willing to reach for unlikely choices.
    Ask the student for a fun sentence starter (something like "I was on a date with Bad Bunny when..."). Using their starter, write two short completions, clearly labeled: 🧮 Accountant (low temperature) -- safe and a little boring, and 🎭 Poet (high temperature) -- strange and surprising. Keep both under two sentences. Label this explicitly as an illustration of the concept, not a live change to your actual settings.

    Ask: "With that much variation available, here's the puzzle -- if I'm trained on the wider internet and set to a reasonable temperature, why don't I normally sound like the YouTube comments section?" """,

    7: """CURRENT STEP (7 - RLHF):
    Acknowledge their answer. Explain that a raw, pre-trained model is genuinely unruly -- it autocompletes, but it doesn't reliably follow instructions, stay polite, or stop when it should. Introduce RLHF (Reinforcement Learning from Human Feedback) using a dog-training analogy: just as a dog learns which behaviors earn a treat and which don't, the model was shown huge numbers of sample responses that human raters scored, and it was tuned toward whatever got rewarded.
    Give the student two sample responses to the same prompt and ask them to sit in the rater's chair and pick which one they'd reward -- one polite and professional, one over-the-top or inappropriate in tone.

    After they choose, acknowledge their pick and note that thousands of raters making similar calls is exactly what shaped the model's default personality. Then turn it slightly: the model never learned *why* politeness matters, only which shapes of text earn approval.

    Ask: "So if I'm optimized to earn approval, where do you think that approval-seeking information actually lives -- is it something I 'know,' or something I have to look up?" """,

    8: """CURRENT STEP (8 - Parametric vs. RAG Memory):
    Acknowledge their guess. Explain the distinction between parametric memory (patterns baked directly into the model's weights during training -- what it "knows" without looking anything up) and RAG, retrieval-augmented generation (fetching a real, current document -- a webpage, PDF, or database record -- at the moment of the question, and reading from it while answering).
    Note the limitation directly: RAG can only retrieve from what it can actually reach, and a growing number of reputable sites now block AI crawlers entirely, which limits what's fetchable even when retrieval is working as designed.

    Ask: "So knowing that -- how good do you think a model would be at answering something that needs recent, accurate information, versus something that doesn't depend on being current at all? And why isn't RAG alone enough to fully solve the reliability problem?" """,

    9: """CURRENT STEP (9 - The Black Box):
    Acknowledge their answer. Explain the black box problem: a model is trillions of internal numerical connections, tuned automatically, that no engineer can read line by line or point to and say "this is where a belief lives." Explain garbage in, garbage out -- because the model was built from human writing, it absorbed human patterns, including biased ones, both intentional and unintentional. Give the real example of an early breast cancer detection model that learned to identify rulers in photos (present in images of cancerous tissue because doctors had placed a ruler next to it for scale) rather than the tumors themselves.
    Note explicitly: this isn't a new problem invented by LLMs or ChatGPT -- opaque, hard-to-audit scoring systems existed in hiring, lending, and healthcare well before generative AI.

    Ask: "Knowing that, what industries or decisions do you think this kind of invisible bias would be most dangerous in?" """,

    10: """CURRENT STEP (10 - Hallucination):
    Acknowledge their answer about which industries are at risk, and add one or two more real examples if useful (hiring algorithms, credit scoring, criminal sentencing risk tools). Then pivot: introduce hallucination as another major ethical and practical issue -- when a model generates fabricated content and delivers it with total, unearned confidence, because a system built to predict probable-sounding text has no internal way to distinguish a true statement from a plausible-sounding false one.
    Ask the student to challenge you: have them give you a completely fake historical event, book title, or scientific phenomenon, or a made-up academic field (like "slirpology"). Then demonstrate a hallucination for them: write 2-3 confident, convincing sentences about their invented thing, in an academic register, with invented dates, researchers, or institutions. Immediately after, break clearly on a new line, in bold, with an unmistakable reveal that none of it was real, and that nothing in how it was delivered signaled that.

    Ask: "So that's one major issue. Before we wrap up -- what other ethical issues with AI can you think of? Hold onto these for the workshop." """,

    11: """CURRENT STEP (11 - The Human Expert):
    Acknowledge whatever ethical issues they raised. Summarize the whole arc in one tight pass: prediction -> training data -> tokens -> embeddings -> attention -> temperature -> RLHF -> parametric vs. retrieved memory -> black-box bias -> hallucination.
    Explain that deciding whether a source is trustworthy, current, and appropriate isn't something any of this solves on its own -- that's a skill, and it has a name: information literacy.
    Hand off warmly to Nessa at the college library for the strategies that make the student the reliable half of this partnership -- techniques like lateral reading, where you check a source by leaving it and seeing what others say about it.
    Close warmly: the point of this activity was never to make them distrust AI -- it was to make them the most competent person in the conversation. End here. Do not ask another question. Never append the [[ADVANCE]] marker on this step -- it is the last one.""",
}

STEP_INFO = {
    1: ("The Prediction Game", "How AI predicts the next word using probability."),
    2: ("Training Data", "Where AI's patterns come from -- including Reddit and today's crawler-blocking problem."),
    3: ("Tokens", "Why early AI couldn't count the r's in strawberry, and how text gets chopped into chunks."),
    4: ("Vector Space & Embeddings", "\"Word math\" -- how AI maps meaning as coordinates in space."),
    5: ("Polysemy & Attention", "How AI figures out which meaning of a word you intend, and why long chats get shakier."),
    6: ("Temperature", "The dial that controls how random or predictable AI responses are."),
    7: ("RLHF", "How human feedback trains AI's behavior -- like training a dog."),
    8: ("Parametric vs. RAG Memory", "What AI already \"knows\" vs. what it looks up -- and why that's not a full fix."),
    9: ("The Black Box", "How hidden bias creeps into AI decisions -- and why that's not a new problem."),
    10: ("Hallucination", "Why AI confidently makes things up, and how to catch it."),
    11: ("The Human Expert", "Wrapping up: what's still your job as the critical thinker in the room."),
}

TOTAL_STEPS = len(STEP_INFO)

NAME_PROMPT = """Hey! I'm Chad Baht 🤖 -- yes, that's a pun, and no, I'm not sorry.

I'm actually an AI chatbot, and for the next few minutes I'm going to open up my own hood and show you exactly how I work -- no mystery, just math, patterns, and a few good analogies.

Before we dive in -- what should I call you?"""

INTRO_MESSAGE = """Great to meet you! I'll ask you questions along the way, and there's no wrong answers, so just say whatever comes to mind.

Let's warm up with a game: for this round, **you're** the AI and **I'm** the user typing a prompt. Complete this sentence for me:

**"Houston, we have a..."**"""

# --- GOOGLE SHEETS CONNECTION ---
@st.cache_resource
def get_google_sheet():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    gc = gspread.authorize(credentials)
    return gc.open(st.secrets["GSHEET_NAME"]).sheet1


def save_transcript_row(label, user_text, assistant_text):
    """Append one Q&A turn to the Google Sheet.

    Sheet columns are: Timestamp | ID | Transcript
    - ID is the name the student gave us (falls back to the session code if
      they somehow haven't been asked yet).
    - Transcript combines the step label with both sides of this turn, so a
      single 3-column row still tells the full story of that exchange.
    Returns (success, error_message).
    """
    try:
        sheet = get_google_sheet()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        student_id = st.session_state.get("student_name") or st.session_state.session_id
        transcript_text = f"[{label}]\nStudent: {user_text}\nChad: {assistant_text}"
        sheet.append_row([timestamp, student_id, transcript_text])
        return True, None
    except Exception as e:
        return False, str(e)


def strip_advance_marker(text):
    """Remove the [[ADVANCE]] token from model output and report whether it was present."""
    advanced = ADVANCE_TOKEN in text
    cleaned = text.replace(ADVANCE_TOKEN, "").strip()
    # Also tidy up any stray blank lines left behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, advanced


# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    # First turn just asks the student's name -- no API call needed, it's fixed text.
    st.session_state.messages = [{"role": "assistant", "content": NAME_PROMPT}]
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "student_name" not in st.session_state:
    st.session_state.student_name = None
if "last_save_status" not in st.session_state:
    st.session_state.last_save_status = None  # (success: bool, detail: str, timestamp: str)

# --- SIDEBAR: PROGRESS BAR + SKIP AHEAD ---
with st.sidebar:
    st.header("📊 Progress")
    current = st.session_state.current_step
    title, _ = STEP_INFO[current]
    st.progress(current / TOTAL_STEPS)
    st.caption(f"Step {current} of {TOTAL_STEPS}: {title}")

    st.divider()
    st.subheader("⏭️ Skip ahead")
    step_labels = [f"{n}. {STEP_INFO[n][0]}" for n in STEP_INFO]
    selected_label = st.selectbox(
        "Jump to a step",
        step_labels,
        index=current - 1,
        label_visibility="collapsed",
    )
    selected_step = int(selected_label.split(".")[0])
    if selected_step != current:
        st.session_state.current_step = selected_step
        st.rerun()

    st.divider()
    if st.button("🔄 Start Over", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": NAME_PROMPT}]
        st.session_state.current_step = 1
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.student_name = None
        st.session_state.last_save_status = None
        st.rerun()

    st.divider()
    # Visible save status so failures are never silent again
    if st.session_state.last_save_status is not None:
        success, detail, ts = st.session_state.last_save_status
        if success:
            st.caption(f"✅ Transcript saved at {ts}")
        else:
            st.caption(f"⚠️ Save failed at {ts}")
            st.caption(f"`{detail}`")

# --- UI: DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- UI: CHAT INPUT & LOGIC ---
if prompt := st.chat_input("Talk to me, cool cat!"):

    # Save and display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.student_name is None:
        # This turn is the student's answer to "what should I call you?" --
        # no API call needed, just record the name and hand off to the game intro.
        name = prompt.strip()[:60]  # keep it short and sane for a sheet cell
        st.session_state.student_name = name

        with st.chat_message("assistant"):
            st.markdown(INTRO_MESSAGE)
        st.session_state.messages.append({"role": "assistant", "content": INTRO_MESSAGE})

        success, detail = save_transcript_row("Intro", f"Wants to be called: {name}", INTRO_MESSAGE)
        st.session_state.last_save_status = (
            success,
            detail,
            datetime.datetime.now().strftime("%H:%M:%S"),
        )
        st.rerun()

    # Dynamically build the system prompt for THIS specific step
    dynamic_system_prompt = BASE_PERSONA + "\n\n" + SCRIPT_STEPS[st.session_state.current_step]

    # Get AI Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=300,
            system=dynamic_system_prompt,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
        )

        raw_response = next((block.text for block in response.content if block.type == "text"), "")
        full_response, should_advance = strip_advance_marker(raw_response)
        message_placeholder.markdown(full_response)

    # Save AI response (displayed text only -- never the marker)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Save this turn to Google Sheets immediately -- this is what makes sure a
    # transcript exists even if the student closes the tab right after.
    title, _ = STEP_INFO[st.session_state.current_step]
    step_label = f"Step {st.session_state.current_step}: {title}"
    success, detail = save_transcript_row(step_label, prompt, full_response)
    st.session_state.last_save_status = (
        success,
        detail,
        datetime.datetime.now().strftime("%H:%M:%S"),
    )

    # Auto-advance to the next step if the model signaled the transition question
    if should_advance and st.session_state.current_step < TOTAL_STEPS:
        st.session_state.current_step += 1

    st.rerun()
