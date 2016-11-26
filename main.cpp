// Copyright 2015 Peter
#include <iostream>
#include <vector>

#include "inc/io.h"
#include "stopwatch/stopwatch.h"

#include "inc/ford-fulkerson.h"
#include "inc/relabel_to_front.h"
#include "inc/residual_network.h"
#include "stanford.cpp"

using namespace std;

auto GetStanfordFlow(ResidualNetwork &E) -> PushRelabel
{
    PushRelabel PR(E.getCount());

    for (int i = 0; i < E.getCount(); ++i)
    {
        for (auto edge : E.getOutgoingEdges(i))
        {
            if (edge.weight > 0)
            {
                PR.AddEdge(edge.from, edge.to, edge.weight);
            }
        }
    }

    return PR;
}

int main()
{
    Stopwatch S;
    S.set_mode(REAL_TIME);
    S.start("TOTAL");

    //
    // Initialization
    //
    S.start("READ");
    ResidualNetwork E = IO::ReadGraph();
    S.stop("READ");

    //
    // RELABEL TO FRONT
    //
    S.start("RTF");
    RelabelToFront RTF(E);
    RTF.Run();
    S.stop("RTF");
    S.stop("TOTAL");


    // //
    // // FORD FULKERSON
    // //
    // S.start("FF");
    // FordFulkerson FF(E);
    // FF.Run();
    // S.stop("FF");
    //
    // //
    // // Stanford
    // //
    // S.start("S");
    // auto PR = GetStanfordFlow(E);
    // long long flowS = PR.GetMaxFlow(E.getSource(), E.getSink());
    // S.stop("S");

    //
    // RESULTS
    //
    cout << "Vertices   : " << E.getCount() << endl;
    cout << "Edges      : " << E.getEdgesCount() << endl;
    // cout << "\n";
    cout << "RTF Flow   : " << RTF.E.getFlow() << endl;
    // cout << "Sta Flow : " << flowS << endl;
    // cout << "FF  Flow : " << FF.E.getFlow() << endl;
    // cout << "\n";
    cout << "READ Time  : " << S.get_total_time("READ") << endl;
    cout << "RTF Time   : " << S.get_total_time("RTF") << endl;
    cout << "TOTAL Time : " << S.get_total_time("TOTAL") << endl;
    // cout << "Sta Time : " << S.get_total_time("S") << endl;
    // cout << "FF  Time : " << S.get_total_time("FF") << endl;
    // cout << "\n";
    // cout << "FF Paths : " << FF.IterationsCount << endl;
    // cout << "\n";
    cout << "Push     : " << RTF.PushCount << endl;
    cout << "Relabel  : " << RTF.RelabelCount << endl;
    cout << "Discharge: " << RTF.DischargeCount << endl;

    cout << "\n";
    // cout << "\n";

    return 0;
}
