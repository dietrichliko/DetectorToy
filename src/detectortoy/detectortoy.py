"""DetectorToy - A tool to study tracking accuracy for FCCee detector.

The python implementation is based on LicToy 2.0
"""
from typing import Tuple
from typing import cast
from typing import Any
import math

import yaml
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float_]
IntArray = npt.NDArray[np.int_]


class Magnet(yaml.YAMLObject):
    """Magnet system.

    Attributes:
        field: Magnetic field strength in T
    """

    yaml_tag = "!magnet"
    field: float

    def __str__(self) -> str:
        """Print Magnet."""
        return f"Magnet(field={self.field})"


class BeamPipe(yaml.YAMLObject):
    """Beampipe.

    Attributes:
        radius: Radius
        x0: Material in interaction length
        color: For drawing
    """

    yaml_tag = "!beampipe"
    radius: float
    x0: float
    wedge: float
    color: str

    def __str__(self) -> str:
        """Print Beampipe."""
        return f"BeamPipe(radius={self.radius},X0={self.x0})"

    def draw(self, ax: Any) -> None:
        """Draw a 2D picture."""
        ax.plot([-400, 400], [self.radius, self.radius], c=self.color)
        ax.plot([-400, 400], [-self.radius, -self.radius], c=self.color)

        tan_wedge = math.tan(self.wedge)
        r = 400 * tan_wedge
        z = self.radius / tan_wedge
        ax.plot([-400, -z], [r, self.radius], c=self.color, ls=":")
        ax.plot([-400, -z], [-r, -self.radius], c=self.color, ls=":")
        ax.plot([400, z], [r, self.radius], c=self.color, ls=":")
        ax.plot([400, z], [-r, -self.radius], c=self.color, ls=":")

    def scan(self, theta: FloatArray) -> Tuple[FloatArray, IntArray]:
        """Scan material.

        Arguments:
            theta: array of polar angles

        Returns:
            material / hits for given theta
        """
        x0 = self.x0 / np.sin(theta)
        return x0, np.zeros_like(theta, dtype=int)


class BarrelDetector(yaml.YAMLObject):
    yaml_tag = "!barrel"
    name: str
    radius: list[float]
    z_pos: list[float]
    x0: list[float]
    active: list[bool]
    sigma_rphi: list[float]
    sigma_z: list[float]
    color: str

    def __str__(self) -> str:
        return f"BarrelDetector(name={self.name}, {len(self.radius)} layers)"

    def draw(self, ax) -> None:

        color = ((self.color if a else "black") for a in self.active)
        for r, z, c in zip(self.radius, self.z_pos, color):
            ax.plot([-z, z], [-r, -r], c=c)
            ax.plot([-z, z], [r, r], c=c)

    def scan(self, theta: FloatArray) -> Tuple[FloatArray, IntArray]:

        x0_sum = np.zeros_like(theta)
        n_sum = np.zeros_like(theta, dtype=int)

        for r, z, x0, a in zip(self.radius, self.z_pos, self.x0, self.active):

            hit = r / np.tan(theta) < z
            x0_sum += np.where(hit, x0 / np.sin(theta), 0)
            if a:
                n_sum += np.where(hit, 1, 0)

        return x0_sum, n_sum


class ForwardDetector(yaml.YAMLObject):
    yaml_tag = "!forward"
    name: str
    z_pos: list[float]
    r_min: list[float]
    r_max: list[float]
    x0: list[float]
    active: list[bool]
    sigma_x: list[float]
    sigma_y: list[float]
    color: str

    def __str__(self) -> str:
        return f"ForwardDetector(name={self.name}, {len(self.z_pos)} layers)"

    def draw(self, ax) -> None:

        color = ((self.color if a else "black") for a in self.active)
        for z, r_min, r_max, c in zip(self.z_pos, self.r_min, self.r_max, color):
            ax.plot([z, z], [r_min, r_max], c=c)
            ax.plot([-z, -z], [r_min, r_max], c=c)
            ax.plot([z, z], [-r_min, -r_max], c=c)
            ax.plot([-z, -z], [-r_min, -r_max], c=c)

    def scan(self, theta: FloatArray) -> Tuple[FloatArray, IntArray]:

        x0_sum = np.zeros_like(theta)
        n_sum = np.zeros_like(theta, dtype=np.int_)

        for z_pos, r_min, r_max, x0, a in zip(
            self.z_pos, self.r_min, self.r_max, self.x0, self.active
        ):
            r = z_pos * np.tan(theta)
            hit = np.logical_and(r_min < r, r < r_max)
            x0_sum += np.where(hit, x0 / np.cos(theta), 0.0)
            if a:
                n_sum += np.where(hit, 1, 0)

        return x0_sum, n_sum


class DetectorToy:

    magnet: Magnet
    beampipe: BeamPipe
    barrel: list[BarrelDetector]
    forward: list[ForwardDetector]

    def load(self, path: str) -> None:

        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.Loader)  # noqa:S506

        self.magnet = cast(Magnet, data["magnet"])
        self.beampipe = cast(BeamPipe, data["beampipe"])
        self.barrel = cast(list[BarrelDetector], data["barrel"])
        self.forward = cast(list[ForwardDetector], data["forward"])

    def draw(self) -> None:

        fig, ax = plt.subplots()
        self.beampipe.draw(ax)
        list(map(lambda x: x.draw(ax), self.barrel))
        list(map(lambda x: x.draw(ax), self.forward))
        plt.show()

    def draw_scan(self) -> None:

        theta = np.linspace(0.01, math.pi / 2, 100)
        x0_beam, _ = self.beampipe.scan(theta)
        x0_barrel: list[FloatArray] = []
        n_barrel: list[IntArray] = []
        for barrel in self.barrel:
            x0, n = barrel.scan(theta)
            x0_barrel.append(x0)
            n_barrel.append(n)
        x0_forward: list[FloatArray] = []
        n_forward: list[IntArray] = []
        for forward in self.forward:
            x0, n = forward.scan(theta)
            x0_forward.append(x0)
            n_forward.append(n)

        fig, (ax1, ax2) = plt.subplots(1, 2)

        c = (
            [self.beampipe.color]
            + [b.color for b in self.barrel]
            + [f.color for f in self.forward]
        )
        y = [x0_beam] + x0_barrel + x0_forward
        ax1.set_xlabel(r"$\theta$")
        ax1.set_ylabel(r"$X^0$")
        ax1.set_xlim([0.0, math.pi / 2])
        ax1.stackplot(theta, y, colors=c, step="mid")

        n = n_barrel + n_forward  # type: ignore
        c = [b.color for b in self.barrel] + [f.color for f in self.forward]
        ax2.set_xlabel(r"$\theta$")
        ax2.set_ylabel("hits")
        ax2.set_xlim([0.0, math.pi / 2])
        ax2.stackplot(theta, n, colors=c, step="mid")

        plt.show()


def main():

    dt = DetectorToy()

    dt.load("cld.yaml")

    dt.draw_scan()
