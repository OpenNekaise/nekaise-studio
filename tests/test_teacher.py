from nekaise_loop.teaching import Revision


def test_teacher_can_author_knowledge_without_a_source_quote_gate():
    row = Revision(id="self-authored", text="A teacher-created explanation", training_text="Exact teacher-chosen sequence", errors=[], evidence=[], use_for_training=True, reason="Teach this concept")
    assert row.use_for_training and row.evidence == []
