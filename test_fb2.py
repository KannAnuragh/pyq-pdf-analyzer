from graph.nodes import separate_stem_and_options

content = """A steady current of 8 mA flows through a wire. The number of electrons passing through a cross-section of the wire in 10 s is: [1 mark] (a) 4.0 \\times 10^{16} (b) 5.0 \\times 10^{17} (c) 1.6 \\times 10^{16} (d) 1.0 \\times 10^{17}"""
print(separate_stem_and_options(content))
