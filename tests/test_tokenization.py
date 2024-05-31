from walkjump.utils import token_string_from_tensor, token_string_to_tensor

from tests.fixtures import aho_alphabet_encoder, aho_sequence  # noqa: F401


def test_token_to_string_tofrom_tensor(aho_alphabet_encoder, aho_sequence):  # noqa: F811
    assert (
        aho_sequence
        == token_string_from_tensor(
            token_string_to_tensor(aho_sequence, aho_alphabet_encoder),
            aho_alphabet_encoder,
            from_logits=False,
        )[0]
    )
    assert (
        aho_sequence
        == token_string_from_tensor(
            token_string_to_tensor(aho_sequence, aho_alphabet_encoder, onehot=True),
            aho_alphabet_encoder,
            from_logits=True,
        )[0]
    )
    print("ok")
