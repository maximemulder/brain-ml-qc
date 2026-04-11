from dataclasses import dataclass


@dataclass
class Link:
    name: str
    url: str


@dataclass
class Institution:
    name: str
    links: list[Link]


# Links extracted from: https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html
ABIDE_1 = [
    Institution(
        name="California Institute of Technology",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4871")]
    ),
    Institution(
        name="Carnegie Mellon University",
        links=[
            Link(name="Scan Data A", url="http://www.nitrc.org/frs/downloadlink.php/4909"),
            Link(name="Scan Data B", url="http://www.nitrc.org/frs/downloadlink.php/4910")
        ]
    ),
    Institution(
        name="Kennedy Krieger Institute",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4881")]
    ),
    Institution(
        name="Ludwig Maximilians University Munich",
        links=[
            Link(name="Scan Data A", url="http://www.nitrc.org/frs/downloadlink.php/4883"),
            Link(name="Scan Data B", url="http://www.nitrc.org/frs/downloadlink.php/4916"),
            Link(name="Scan Data C", url="http://www.nitrc.org/frs/downloadlink.php/4917"),
            Link(name="Scan Data D", url="http://www.nitrc.org/frs/downloadlink.php/4918")
        ]
    ),
    Institution(
        name="NYU Langone Medical Center",
        links=[
            Link(name="Scan Data A", url="http://www.nitrc.org/frs/downloadlink.php/4972"),
            Link(name="Scan Data B", url="http://www.nitrc.org/frs/downloadlink.php/4973"),
            Link(name="Scan Data C", url="http://www.nitrc.org/frs/downloadlink.php/4974"),
            Link(name="Scan Data D", url="http://www.nitrc.org/frs/downloadlink.php/4975"),
            Link(name="Scan Data E", url="http://www.nitrc.org/frs/downloadlink.php/4976")
        ]
    ),
    Institution(
        name="Olin, Institute of Living at Hartford Hospital",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4885")]
    ),
    Institution(
        name="Oregon Health and Science University",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4893")]
    ),
    Institution(
        name="San Diego State University",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4903")]
    ),
    Institution(
        name="Social Brain Lab BCN NIC UMC Groningen and Netherlands Institute for Neurosciences",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4905")]
    ),
    Institution(
        name="Stanford University",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4875")]
    ),
    Institution(
        name="Trinity Centre for Health Sciences",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4879")]
    ),
    Institution(
        name="University of California, Los Angeles: Sample 1",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4901")]
    ),
    Institution(
        name="University of California, Los Angeles: Sample 2",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4899")]
    ),
    Institution(
        name="University of Leuven: Sample 1",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4889")]
    ),
    Institution(
        name="University of Leuven: Sample 2",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4891")]
    ),
    Institution(
        name="University of Michigan: Sample 1",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4895")]
    ),
    Institution(
        name="University of Michigan: Sample 2",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4897")]
    ),
    Institution(
        name="University of Pittsburgh School of Medicine",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4907")]
    ),
    Institution(
        name="University of Utah School of Medicine",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4887")]
    ),
    Institution(
        name="Yale Child Study Center",
        links=[Link(name="Scan Data", url="http://www.nitrc.org/frs/downloadlink.php/4873")]
    ),
]

# Links extracted from: https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html
ABIDE_2 = [
    Institution(
        name="Barrow Neurological Institute",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9066")]
    ),
    Institution(
        name="Erasmus University Medical CenterRotterdam",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9064")]
    ),
    Institution(
        name="ETH Zürich",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9090")]
    ),
    Institution(
        name="Georgetown University",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9068")]
    ),
    Institution(
        name="Indiana University",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9070")]
    ),
    Institution(
        name="Institut Pasteur and Robert Debré Hospital",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9072")]
    ),
    Institution(
        name="Katholieke Universiteit Leuven",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9086")]
    ),
    Institution(
        name="Kennedy Krieger Institute",
        links=[
            Link(name="Scan Data A", url="https://www.nitrc.org/frs/downloadlink.php/9098"),
            Link(name="Scan Data B", url="https://www.nitrc.org/frs/downloadlink.php/9100"),
            Link(name="Scan Data C", url="https://www.nitrc.org/frs/downloadlink.php/9101"),
            Link(name="Scan Data D", url="https://www.nitrc.org/frs/downloadlink.php/9102")
        ]
    ),
    Institution(
        name="NYU Langone Medical Center:Sample 1",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9074")]
    ),
    Institution(
        name="NYU Langone Medical Center:Sample 2",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9062")]
    ),
    Institution(
        name="Olin Neuropsychiatry Research Center, Institute of Living at Hartford Hospital",
        links=[
            Link(name="Scan Data A", url="https://www.nitrc.org/frs/downloadlink.php/9104"),
            Link(name="Scan Data B", url="https://www.nitrc.org/frs/downloadlink.php/9105"),
            Link(name="Scan Data C", url="https://www.nitrc.org/frs/downloadlink.php/9106"),
            Link(name="Scan Data D", url="https://www.nitrc.org/frs/downloadlink.php/9107")
        ]
    ),
    Institution(
        name="Oregon Health and Science University",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9076")]
    ),
    Institution(
        name="Trinity Centre for Health Sciences",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9080")]
    ),
    Institution(
        name="San Diego State University",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9078")]
    ),
    Institution(
        name="Stanford University",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9937")]
    ),
    Institution(
        name="University of California Davis",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9082")]
    ),
    Institution(
        name="University of California Los Angeles",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9084")]
    ),
    Institution(
        name="University of Miami",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9935")]
    ),
    Institution(
        name="University of Utah School of Medicine",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9088")]
    ),
    Institution(
        name="University of California Los Angeles: Longitudinal Sample",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9093")]
    ),
    Institution(
        name="University of Pittsburgh School of Medicine: Longitudinal Sample",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9095")]
    )
]
