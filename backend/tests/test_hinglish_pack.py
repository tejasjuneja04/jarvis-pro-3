import hinglish_pack as H


def test_relevant_examples_come_first():
    block = H.examples("mujhe bahut tension ho rahi hai", "Tejas")
    assert "tension" in block.lower() and "Tejas" in block and "{t}" not in block


def test_every_pair_is_well_formed():
    assert len(H.PACK) >= 40
    for user, reply in H.PACK:
        assert user.strip() and reply.strip()
