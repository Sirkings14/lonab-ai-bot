"""Regression tests: every row below is a REAL row that made the parser reject a whole race day."""
from pdf_parser import parse_race_card

def one(row, n, disc):
    """Parse a single real row. The field size is the row's own number and every other slot is
    padded as a scratched horse (N.P.), which the parser skips."""
    num = int(row.split()[0])
    pad = [f"{i:02d} X N.P. N.P. N.P. N.P. N.P. N.P." for i in range(1, num)]
    rows = parse_race_card("\n".join(pad + [row]) + "\n", num, discipline=disc)
    assert len(rows) == 1 and rows[0]["Num"] == f"{num:02d}", f"row not parsed: {row}"
    return rows[0]

def test_attele_odds_missing_slash_one():
    r = one("1 LOULIANA DE VANDEL R. MARTY J.M. BAUDOUIN EC. J.M BAUDOUIN F.5 2850.M 1.12.50 6.4.D.3.3 70 670 60/ 63/1", 1, "ATTELE")
    assert r["Horse"] == "LOULIANA DE VANDEL" and r["Decimal_Odds"] == 61.0 and r["Decimal_Odds2"] == 64.0

def test_attele_empty_perf_slot():
    r = one("5 IN RACE LORRAINE G.A. POU POU G.A. POU POU J.J.R. GUERIN F.8 2850.M 1.10.80 0..0.3.3 205 130 87/1 90/1", 1, "ATTELE")
    assert r["Horse"] == "IN RACE LORRAINE" and r["Earnings"] == 205130.0

def test_attele_two_digit_minute():
    r = one("14 GUILI GUILI D. THOMAIN F. TERRY Y.M. DESAUNETTE H.10 2850.M 11.13.20 4.7.0.6.D 189 700 27/1 25/1", 1, "ATTELE")
    assert r["Horse"] == "GUILI GUILI" and r["Decimal_Odds"] == 28.0

def test_attele_one_digit_hundredths():
    r = one("4 HURRICANE CARTER Y. LEBOURGEOIS S. RAIMOND S. RAIMOND H.9 2850.M 1.09.7 0.2.D.7.3 446 120 50/1 53/1", 1, "ATTELE")
    assert r["Horse"] == "HURRICANE CARTER" and r["Decimal_Odds"] == 51.0

def test_plat_lowercase_kg_with_draw():
    r = one("01 BAYAMOON F.VERON J.REYNIER UNDER BLUE SKY M.5 7 60,5.kg 2.0.8.1.1 103 244 13/1 10/1", 1, "PLAT")
    assert r["Horse"] == "BAYAMOON" and r["Weight_KG"] == 60.5 and r["Draw"] == 7

def test_plat_lowercase_kg_no_draw():
    r = one("01 LOUMINOS N.FERREIRA R.SCHMIDLIN R.SCHMIDLIN H.4 70.kg A.8.7.1.0 134 680 12/1 9/1", 1, "PLAT")
    assert r["Horse"] == "LOUMINOS" and r["Weight_KG"] == 70.0 and r["Perf"] == "A.8.7.1.0"

def test_plat_second_failing_row_shapes():
    one("02 WINDFLOWER H.BESNIER P.COTTIER EC. GRIBOMONT/EC. DU SUD F.4 10 60.kg 4.7.5.2.1 68 314 12/1 9/1", 1, "PLAT")
    one("02 HELL STORM T.BEAURAIN Y.FOUIN BIVANS SRL M.4 70.kg 4.4.4.1.4 52 090 13/1 10/1", 1, "PLAT")

def test_old_shapes_still_parse():
    r = one("01 HAMANDIO J.REVELEY N. GEORGE & A. ZETTERHOLM C.ALEXANDER H.9 72.KG 3.T.2.8.5 347 012 7/1 6/1", 1, "PLAT")
    assert r["Horse"] == "HAMANDIO" and r["Weight_KG"] == 72.0 and r["Decimal_Odds"] == 8.0

def test_scratch_does_not_sink_race():
    txt = ("01 KOHAKOU H.BOUTIN S.GAVILAN HA.FER.MARQUES H.3 7 60.KG 1.1.3.9.3 58 716 31/1 33/1\n"
           "02 CAUDRY N.P. N.P. N.P. N.P. N.P. N.P. N.P. N.P. N.P.\n")
    assert len(parse_race_card(txt, 2)) == 1

def test_merged_line_with_trailing_odds_still_parses():
    r = one("02 AMY DU KIFF T.CHEVILLARD GAB.LEENDERS EC.NOSAT/EC.KERISEL H.7 68.kg 3.3.6.3.3 576 960 19/1 21/1 10/1 21/1 18/1", 2, "PLAT")
    assert r["Horse"] == "AMY DU KIFF" and r["Decimal_Odds"] == 20.0

def test_sunday_flat_rows_from_08_11():
    one("01 DON'T SHUT ME DOWN M.GUYON P&J.BRANDT P.BRANDT H.4 9 61.kg 6.5.1.5.7 85 136 6/1 4/1", 1, "PLAT")
    one("03 BLACK SAXON A.MADAMET N.BELLANGER D.MARS H.5 8 59,5.kg 0.2.4.5.0 94 444 28/1 25/1", 3, "PLAT")

def test_one_unreadable_row_no_longer_cancels_the_race():
    good = "\n".join(f"{i:02d} HORSE{i} J.JOCKEY T.TRAINER OWN H.5 {5+i} 60.kg 1.2.3.4.5 100 000 {4+i}/1 {3+i}/1" for i in range(1, 11) if i != 3)
    bad = "03 ??? garbled line with no usable tail"
    rows = parse_race_card(good + "\n" + bad + "\n", 10, discipline="PLAT")
    assert len(rows) == 9
