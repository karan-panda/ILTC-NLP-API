from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2)
sym_spell.load_dictionary("ILTC_dictionary.txt", term_index=0, count_index=1)

text = input("Enter your text: ")

corrected_text = sym_spell.lookup_compound(text, max_edit_distance=2)
final_text = corrected_text[0].term if corrected_text else text
print(final_text)