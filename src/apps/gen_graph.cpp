#include <iostream>
#include <stdexcept>
#include <string>

#include "sssp/generators.hpp"
#include "sssp/graph.hpp"

using namespace sssp;

namespace {

void usage() {
    std::cout << R"(usage: gen_graph --topo T --n N [options] --out PATH.bin

  --topo T            uniform | rmat | grid | geometric | chain | star | layered | hub
  --n N               vertices (grid: rounded to a square unless --rows/--cols)
  --m M               undirected edges requested (uniform, rmat, geometric, hub)
  --rows R --cols C   grid shape
  --layer-width W     layered: vertices per layer
  --layer-links K     layered: distinct next-layer targets per vertex (default 4)
  --hub-fraction H    hub: share of the m edges attached to one vertex
  --wdist D           unit | uniform | logunif (default uniform)
  --wmin A --wmax B   weight range (default [1,100])
  --decades D         logunif over [1, 10^D]
  --seed S            topology seed (default 1)
  --wseed S           weight seed (default: derived from --seed)
  --rseed S           relabeling seed (default: derived from --seed)
  --no-relabel        keep the generator's natural vertex order
  --directed          emit each edge once instead of both ways
  --out PATH          .bin writes CSR with provenance; anything else a text edge list

Weights come from their own random stream, so changing --wdist, --wmin/--wmax
or --decades leaves the topology bit-identical (same topo_hash). No topology
gets a spanning tree; the bench samples sources from the giant component.
)";
}

}  // namespace

int main(int argc, char** argv) {
    GenSpec s;
    std::string out;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if (a == "-h" || a == "--help") {
                usage();
                return 0;
            } else if (a == "--topo") s.topology = next();
            else if (a == "--n") s.n = std::stoll(next());
            else if (a == "--m") s.m = std::stoll(next());
            else if (a == "--rows") s.rows = std::stoi(next());
            else if (a == "--cols") s.cols = std::stoi(next());
            else if (a == "--layer-width") s.layer_width = std::stoi(next());
            else if (a == "--layer-links") s.layer_links = std::stoi(next());
            else if (a == "--hub-fraction") s.hub_fraction = std::stod(next());
            else if (a == "--wdist") s.wdist = next();
            else if (a == "--wmin") s.wmin = std::stoi(next());
            else if (a == "--wmax") s.wmax = std::stoi(next());
            else if (a == "--decades") s.decades = std::stod(next());
            else if (a == "--seed") s.seed = std::stoull(next());
            else if (a == "--wseed") s.wseed = std::stoull(next());
            else if (a == "--rseed") s.rseed = std::stoull(next());
            else if (a == "--no-relabel") s.relabel = false;
            else if (a == "--directed") s.directed = true;
            else if (a == "--out") out = next();
            else throw std::runtime_error("unknown option " + a);
        }
        if (out.empty()) throw std::runtime_error("--out is required");

        Graph g = generate(s);
        save_graph(out, g);
        std::cerr << "wrote " << out << "  topo=" << s.topology << " n=" << g.n
                  << " arcs=" << g.num_edges() << " topo_hash=" << hex64(topo_hash(g))
                  << " graph_id=" << hex64(graph_hash(g)) << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        usage();
        return 1;
    }
    return 0;
}
