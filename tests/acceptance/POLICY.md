# Decisions behind the frozen acceptance set

The scenario rounds kept trading some cases back and forth. These are the decisions; a case that follows them is
expected to pass, and changing one means changing this file on purpose.

## Pronoun cues

1. When a line reads two ways that change who is older or who is speaking, the expected cue is none (`null`).
   A missing cue only loses a hint; a wrong one makes the model invert the pair. So: "honey" alone, 할머니가 보고
   싶었는지, a SIS tab, and spouse calls with nothing showing which one is the husband (여보, babe, 자기야) give no
   direction (`null` or `["", ""]`). A junior of unknown gender keeps their own word blank.
2. 어머님/아버님/お母さん/妈 in an ordinary line give the family cue (con + mẹ/bố): a child or in-law says them far
   more often than staff do, and this holds for a doctor's plain "어머님, 걱정 마세요" too. Only phrases staff alone
   use (検査しましょう, 出しておきます, 검사 결과, 수업, 상담, 담임, 三者面談, 您家孩子, 开点药, お子さん…) give no
   cue, and so does a suitor meeting the parent (처음 뵙겠습니다, 남자친구입니다, お嬢さんを僕にください).
3. A doctor called 선생님/先生 next to clinic words (病院, 병원, 입원, 熱…), and a parent asking a teacher about their
   own child, give no teacher/pupil cue.
4. A family title before 我 with 你 later (姐姐我好想你, 哥我跟你说, 爸爸我给你买了) gives no cue: either side may say
   it. Glued to 我 with no 你 (姐姐我可是学霸, 爷爷我当年…), the title is the speaker naming themself: the elder's cue.
   A title called and then claimed by the speaker (姐姐,我做姐姐的…) gives none.
5. "kiddo" gives no cue (any elder says it); "son" gives the parent cue unless it starts a list of names or is a
   gamer's taunt or hype ("owned, son", "let's go, son").
6. Words for the whole audience (大家, 家人们, 여러분, everyone) give no pair.
7. A speaker's own word for themself (臣妾, 儿臣, 晚辈, 本宫) sets their side and stays set across pages.
8. Family words use the Vietnamese for that side of the family: 舅舅/외삼촌 cậu, 伯伯/伯父/큰아버지 bác, 이모/姨妈 dì,
   姑姑/고모 cô, 叔叔/삼촌 chú.

## Layout

9. A title centred over a line in a speech bubble or subtitle is a call and joins it with a comma ("Mom, help!"),
   even when the line turns out to be that person's own (the text cannot show colour). A title flush left over a
   wider line in a dialogue box is the speaker's name tab. A title before a narrative verb (NEVER, CAME, 总是说,
   주셨다…) is the subject of the sentence; this holds for the text alone too, so such a title is neither a speaker
   nor a call.
10. Three or more short unpunctuated rows lined up at menu spacing are separate items. Rows that run on into a row
    ending in punctuation are one speech bubble. All-caps rows centred and close together are comic lettering.
11. In area mode, short choices stacked under a question, or known choice words (はい/いいえ, 是/否, Yes/No), stay
    apart; other short two-row subtitles are one line.
12. Furigana, HUD buttons on a name row (AUTO SKIP LOG), corner tags (LIVE, TV-14, S2 E5, HD, 第12集), headings
    (OBJECTIVE, MEANWHILE), panel, popup, tracker and toast titles (Side Quest, 報酬, 파티 초대, 実績解除), time
    captions (三年后, NEXT MORNING, 그날밤) and interjections (응, 嗯, 헐) are never a speaker. Corner tags are dropped.
13. Reading order across manga panels with no visible border, and repairing OCR look-alikes (エ学, 38。), are not
    required.

## Voice and text

14. Numbers are read the way a Vietnamese speaker says them; a form the reader cannot say confidently is read
    digit by digit rather than wrongly.
15. All-caps text reaches the models in sentence case, keeping buttons, keys, acronyms and gamer tags (RT, [E], HP,
    MP5, START, XxSNIPERxX).

## Where the snapshot gave way

The 165-screen snapshot recorded the behaviour of rc55. Four screens recorded what the rounds reported as a bug and
follow the cases instead: battle_feed_en (one block per toast), battle_hud_en (every party member treated the same),
backlog_ja (the backlog title alone) and panel_headers_en/ja (panel headers are not name tabs).
