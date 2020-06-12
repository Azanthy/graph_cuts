#include "graph.hh"
#include "gpu.hh"
#include <vector>
#include <benchmark/benchmark.h>

void BM_Rendering_cpu(benchmark::State& st)
{
    auto graph = Graph("segmentation_dataset/inputs/normal/124084.jpg",
            "segmentation_dataset/inputs/marked/marked_124084.jpg",
            st.range(0));

    for (auto _ : st)
        graph.max_flow();

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_Rendering_gpu(benchmark::State& st)
{
    auto graph = Graph("segmentation_dataset/inputs/normal/124084.jpg",
            "segmentation_dataset/inputs/marked/marked_124084.jpg",
            st.range(0));

    for (auto _ : st)
        max_flow_gpu(graph);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_Rendering_cpu)->Arg(50)->Arg(100)->Arg(400)->Arg(1000)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Rendering_gpu)->Arg(50)->Arg(100)->Arg(400)->Arg(1000)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_MAIN();
