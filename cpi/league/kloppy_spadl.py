"""kloppy event datasets to SPADL actions, for providers without a native socceraction converter.

socceraction 1.5.3 ships a kloppy converter, but it was written for kloppy 3.15 and breaks on kloppy 3.16+ in two ways:
  - the coordinate-system classes changed, so the converter fails to build its target system;
  - kloppy's Orientation.HOME_AWAY now switches sides at half-time, while SPADL keeps the home team attacking left to
    right for the whole match (kloppy's STATIC_HOME_AWAY). Left as is, the second half of every match is mirrored.
This module fixes both (an equivalent SPADL coordinate system from kloppy's CustomCoordinateSystem: origin bottom left,
y up, 105 x 68 metres; and the static orientation) and otherwise uses socceraction's code unchanged.

socceraction only validates this route for StatsBomb. Other providers (Wyscout v3, Opta, Sportec, Metrica, ...) convert,
but event definitions differ between providers, so their results are flagged as experimental in the report.
"""
import warnings
from types import SimpleNamespace

from socceraction import spadl


def _patch():
    import socceraction.spadl.kloppy as SK
    from kloppy.domain import CustomCoordinateSystem, Dimension, MetricPitchDimensions, Orientation, Origin, VerticalOrientation

    def spadl_coordinates(pitch_length=None, pitch_width=None):
        return CustomCoordinateSystem(
            origin=Origin.BOTTOM_LEFT, vertical_orientation=VerticalOrientation.BOTTOM_TO_TOP,
            pitch_dimensions=MetricPitchDimensions(x_dim=Dimension(0, spadl.config.field_length), y_dim=Dimension(0, spadl.config.field_width),
                                                   pitch_length=pitch_length or 105, pitch_width=pitch_width or 68, standardized=True))
    SK._SoccerActionCoordinateSystem = spadl_coordinates
    SK.Orientation = SimpleNamespace(HOME_AWAY=Orientation.STATIC_HOME_AWAY)     # the converter only reads HOME_AWAY
    return SK


def convert(dataset, game_id):
    """SPADL actions (with names) for one kloppy EventDataset, all actions in the home team's direction."""
    SK = _patch()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        actions = SK.convert_to_actions(dataset, game_id=game_id)
    return spadl.add_names(actions)
