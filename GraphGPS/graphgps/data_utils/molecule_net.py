from torch_geometric.datasets import MoleculeNet


class MyMoleculeNet(MoleculeNet):

    names = {
        'esol': ['ESOL', 'delaney-processed.csv', 'delaney-processed', -1, -2],
        'freesolv': ['FreeSolv', 'SAMPL.csv', 'SAMPL', 1, 2],
        'lipo': ['Lipophilicity', 'Lipophilicity.csv', 'Lipophilicity', 2, 1],
        'pcba': ['PCBA', 'pcba.csv.gz', 'pcba', -1,
                 slice(0, 128)],
        'muv': ['MUV', 'muv.csv.gz', 'muv', -1,
                slice(0, 17)],
        'hiv': ['HIV', 'HIV.csv', 'HIV', 0, -1],
        'bace': ['BACE', 'bace.csv', 'bace', 0, 2],
        'bbbp': ['BBPB', 'BBBP.csv', 'BBBP', -1, -2],
        'tox21': ['Tox21', 'tox21.csv.gz', 'tox21', -1,
                  slice(0, 12)],
        'toxcast':
        ['ToxCast', 'toxcast_data.csv.gz', 'toxcast_data', 0,
         slice(1, 618)],
        'sider': ['SIDER', 'sider.csv.gz', 'sider', 0,
                  slice(1, 28)],
        'clintox': ['ClinTox', 'clintox.csv.gz', 'clintox', 0,
                    slice(1, 3)],
        'metstab': ['Metstab', 'metstab.csv.gz', 'metstab', 0,
                    slice(1, 3)],
        'estrogen': ['Estrogen', 'estrogen.csv.gz', 'estrogen', 0,
                    slice(1, 3)],
    }

    def __init__(self, root: str, name: str):
        super().__init__(root, name)
        self.names['metstab'] = ['Metstab', 'metstab.csv.gz', 'metstab', 0, slice(1, 3)]
        self.names['estrogen'] = ['Estrogen', 'estrogen.csv.gz', 'estrogen', 0, slice(1, 3)]