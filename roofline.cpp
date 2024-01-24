/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <iostream>
#include <chrono>

#include <occa.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/testing.hpp>
//======================================

occa::json parseArgs(int argc, const char **argv) {
  occa::cli::parser parser;
  parser
    .withDescription(
      "Empirical Roofline Benchmark"
    )
    .addOption(
      occa::cli::option('d', "device",
                        "Device mode. Can be Serial, OpenMP, OpenCL, CUDA, HIP, or SYCL (default: Serial)")
      .withArg()
      .withDefaultValue("Serial")
    )
    .addOption(
      occa::cli::option('n', "entries",
                        "Vector length")
      .withArg()
      .withDefaultValue("134217728")
    )
    .addOption(
      occa::cli::option('t', "type",
                        "Compute type. Can be fp64 or fp32 (default: fp64)")
      .withArg()
      .withDefaultValue("fp64")
    );

  occa::json args = parser.parseArgs(argc, argv);
  return args;
}

using dfloat=float;

using timePoint_t = std::chrono::time_point<std::chrono::high_resolution_clock>;

template<typename T>
void run(occa::json &args) {
  std::string mode;
  if (args["options/device"]=="Serial") {
    mode = "{mode: 'Serial'}";
  } else if (args["options/device"]=="OpenMP") {
    mode = "{mode: 'OpenMP'}";
  } else if (args["options/device"]=="OpenCL") {
    mode = "{mode: 'OpenCL', platform_id: 0, device_id: 0}";
  } else if (args["options/device"]=="CUDA") {
    mode = "{mode: 'CUDA', device_id: 0}";
  } else if (args["options/device"]=="HIP") {
    mode = "{mode: 'HIP', device_id: 0}";
  } else if (args["options/device"]=="SYCL") {
    mode = "{mode: 'SYCL', device_id: 0}";
  }

  occa::device device(mode);

  std::string cacheDir = std::string(ROOFLINE_DIR) + "/.occa";
  occa::env::setOccaCacheDir(cacheDir);

  // OCCA build stuff
  occa::json props; //copy base occa properties

  props["defines"].asObject();
  props["serial/include_std"] = true;

  if(sizeof(T)==4){
    props["defines/dfloat"] = "float";
  }
  if(sizeof(T)==8){
    props["defines/dfloat"] = "double";
  }

  props["compiler_flags"] += "-O3 ";

  std::string arch = device.arch();
  if (device.mode()=="HIP") {
    arch = arch.substr(0,arch.find(":")); //For HIP mode, remove the stuff after the :
  }

  bool hasMFMAs = false;
  if (sizeof(T)==8 && device.mode()=="HIP" &&
      (arch=="gfx90a" || arch=="gfx940" || arch=="gfx941" || arch=="gfx942")) {
    props["defines/HAS_FP64_MFMA"] = 1;
    hasMFMAs = true;
  }
  if (sizeof(T)==4 && device.mode()=="HIP" &&
      (arch=="gfx90a" || arch=="gfx940" || arch=="gfx941" || arch=="gfx942")) {
    props["defines/HAS_FP32_MFMA"] = 1;
    hasMFMAs = true;
  }

  std::string kernelFileName = std::string(ROOFLINE_DIR) + "/roofline.okl";

  occa::kernel fillKernel = device.buildKernel(
                                    kernelFileName,
                                    "fillKernel",
                                    props
                                   );

  occa::kernel fmaKernel  = device.buildKernel(
                                    kernelFileName,
                                    "fmaRate",
                                    props
                                   );

  occa::kernel shmem1Kernel  = device.buildKernel(
                                    kernelFileName,
                                    "shmem1Rate",
                                    props
                                   );

  occa::kernel shmem2Kernel  = device.buildKernel(
                                    kernelFileName,
                                    "shmem2Rate",
                                    props
                                   );

  occa::kernel mfma4Kernel  = device.buildKernel(
                                    kernelFileName,
                                    "mfma4Rate",
                                    props
                                   );

  occa::kernel shmem1mfma4Kernel  = device.buildKernel(
                                    kernelFileName,
                                    "shmem1mfma4Rate",
                                    props
                                   );

  occa::kernel shmem2mfma4Kernel  = device.buildKernel(
                                    kernelFileName,
                                    "shmem2mfma4Rate",
                                    props
                                   );

  occa::kernel mfma16Kernel  = device.buildKernel(
                                    kernelFileName,
                                    "mfma16Rate",
                                    props
                                   );

  occa::kernel shmem1mfma16Kernel  = device.buildKernel(
                                    kernelFileName,
                                    "shmem1mfma16Rate",
                                    props
                                   );

  occa::kernel shmem2mfma16Kernel  = device.buildKernel(
                                    kernelFileName,
                                    "shmem2mfma16Rate",
                                    props
                                   );

  //create array buffers
  const int N = std::stoi(args["options/entries"]);

  occa::memory o_x = device.malloc<T>(N);
  occa::memory o_y = device.malloc<T>(N);

  T a = 0.5;

  {
    int Nwarm = 5;
    for(int n=0;n<Nwarm;++n){ //warmup
      fmaKernel(N, 8, a, o_x, o_y);
    }
    device.finish();

    //test
    for(int k=1;k<=4096;k*=2){

      timePoint_t start = std::chrono::high_resolution_clock::now();;

      /* COPY Test */
      int Ntests = 20;
      for(int n=0;n<Ntests;++n){
        fmaKernel(N, k, a, o_x, o_y);
      }

      device.finish();
      timePoint_t end = std::chrono::high_resolution_clock::now();;
      double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/(1.0e6)/Ntests;

      size_t bytesIn  = N*sizeof(T);
      size_t bytesOut = N*sizeof(T);
      size_t bytes = bytesIn + bytesOut;

      size_t flops = 2*k*static_cast<size_t>(N);

      printf("FMA:        BW=%8.2f GB/s, AI=%8.2f FLOP/B, GFLOPS=%8.2f\n",
              (static_cast<double>(bytes)/1.e9)/elapsedTime, static_cast<double>(flops)/bytes, (static_cast<double>(flops)/1.e9)/elapsedTime);
    }
  }

  {
    int Nwarm = 5;
    for(int n=0;n<Nwarm;++n){ //warmup
      shmem1Kernel(N, 8, a, o_x, o_y);
    }
    device.finish();

    //test
    for(int k=1;k<=4096;k*=2){
      timePoint_t start = std::chrono::high_resolution_clock::now();;

      /* COPY Test */
      int Ntests = 20;
      for(int n=0;n<Ntests;++n){
        shmem1Kernel(N, k, a, o_x, o_y);
      }

      device.finish();
      timePoint_t end = std::chrono::high_resolution_clock::now();;
      double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/(1.0e6)/Ntests;

      size_t bytesIn  = N*sizeof(T);
      size_t bytesOut = N*sizeof(T);
      size_t bytes = bytesIn + bytesOut;

      size_t flops = 2*k*static_cast<size_t>(N);

      printf("FMA SHMEM1: BW=%8.2f GB/s, AI=%8.2f FLOP/B, GFLOPS=%8.2f, SHMEM AI=%8.2f FLOP/B\n",
              (static_cast<double>(bytes)/1.e9)/elapsedTime, static_cast<double>(flops)/bytes, (static_cast<double>(flops)/1.e9)/elapsedTime, static_cast<double>(2)/sizeof(T));
    }
  }

  {
    int Nwarm = 5;
    for(int n=0;n<Nwarm;++n){ //warmup
      shmem2Kernel(N, 8, a, o_x, o_y);
    }
    device.finish();

    //test
    for(int k=1;k<=4096;k*=2){
      timePoint_t start = std::chrono::high_resolution_clock::now();;

      /* COPY Test */
      int Ntests = 20;
      for(int n=0;n<Ntests;++n){
        shmem2Kernel(N, k, a, o_x, o_y);
      }

      device.finish();
      timePoint_t end = std::chrono::high_resolution_clock::now();;
      double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/(1.0e6)/Ntests;

      size_t bytesIn  = N*sizeof(T);
      size_t bytesOut = N*sizeof(T);
      size_t bytes = bytesIn + bytesOut;

      size_t flops = 2*k*static_cast<size_t>(N);

      printf("FMA SHMEM2: BW=%8.2f GB/s, AI=%8.2f FLOP/B, GFLOPS=%8.2f, SHMEM AI=%8.2f FLOP/B\n",
              (static_cast<double>(bytes)/1.e9)/elapsedTime, static_cast<double>(flops)/bytes, (static_cast<double>(flops)/1.e9)/elapsedTime, static_cast<double>(2)/(2*sizeof(T)));
    }
  }

  if (hasMFMAs) {
    int Nwarm = 5;
    for(int n=0;n<Nwarm;++n){ //warmup
      mfma4Kernel(N, 8, a, o_x, o_y);
    }
    device.finish();

    //test
    for(int k=1;k<=1024;k*=2){

      timePoint_t start = std::chrono::high_resolution_clock::now();;

      /* COPY Test */
      int Ntests = 20;
      for(int n=0;n<Ntests;++n){
        mfma4Kernel(N, k, a, o_x, o_y);
      }

      device.finish();
      timePoint_t end = std::chrono::high_resolution_clock::now();;
      double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/(1.0e6)/Ntests;

      size_t bytesIn  = N*sizeof(T);
      size_t bytesOut = N*sizeof(T);
      size_t bytes = bytesIn + bytesOut;

      size_t flops = (512/64)*k*static_cast<size_t>(N);

      printf("MFMA4:        BW=%8.2f GB/s, AI=%8.2f FLOP/B, GFLOPS=%8.2f, SHMEM AI=%8.2f FLOP/B\n",
              (static_cast<double>(bytes)/1.e9)/elapsedTime, static_cast<double>(flops)/bytes, (static_cast<double>(flops)/1.e9)/elapsedTime, static_cast<double>(512)/(64*sizeof(T)));
    }
  }

  if (hasMFMAs) {
    int Nwarm = 5;
    for(int n=0;n<Nwarm;++n){ //warmup
      shmem1mfma4Kernel(N, 8, a, o_x, o_y);
    }
    device.finish();

    //test
    for(int k=1;k<=1024;k*=2){
      timePoint_t start = std::chrono::high_resolution_clock::now();;

      /* COPY Test */
      int Ntests = 20;
      for(int n=0;n<Ntests;++n){
        shmem1mfma4Kernel(N, k, a, o_x, o_y);
      }

      device.finish();
      timePoint_t end = std::chrono::high_resolution_clock::now();;
      double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/(1.0e6)/Ntests;

      size_t bytesIn  = N*sizeof(T);
      size_t bytesOut = N*sizeof(T);
      size_t bytes = bytesIn + bytesOut;

      size_t flops = (512/64)*k*static_cast<size_t>(N);

      printf("MFMA4 SHMEM1: BW=%8.2f GB/s, AI=%8.2f FLOP/B, GFLOPS=%8.2f, SHMEM AI=%8.2f FLOP/B\n",
              (static_cast<double>(bytes)/1.e9)/elapsedTime, static_cast<double>(flops)/bytes, (static_cast<double>(flops)/1.e9)/elapsedTime, static_cast<double>(512)/(64*sizeof(T)));
    }
  }

  if (hasMFMAs) {
    int Nwarm = 5;
    for(int n=0;n<Nwarm;++n){ //warmup
      shmem2mfma4Kernel(N, 8, a, o_x, o_y);
    }
    device.finish();

    //test
    for(int k=1;k<=1024;k*=2){
      timePoint_t start = std::chrono::high_resolution_clock::now();;

      /* COPY Test */
      int Ntests = 20;
      for(int n=0;n<Ntests;++n){
        shmem2mfma4Kernel(N, k, a, o_x, o_y);
      }

      device.finish();
      timePoint_t end = std::chrono::high_resolution_clock::now();;
      double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/(1.0e6)/Ntests;

      size_t bytesIn  = N*sizeof(T);
      size_t bytesOut = N*sizeof(T);
      size_t bytes = bytesIn + bytesOut;

      size_t flops = (512/64)*k*static_cast<size_t>(N);

      printf("MFMA4 SHMEM2: BW=%8.2f GB/s, AI=%8.2f FLOP/B, GFLOPS=%8.2f, SHMEM AI=%8.2f FLOP/B\n",
              (static_cast<double>(bytes)/1.e9)/elapsedTime, static_cast<double>(flops)/bytes, (static_cast<double>(flops)/1.e9)/elapsedTime, static_cast<double>(512)/(2*64*sizeof(T)));
    }
  }

  if (hasMFMAs) {
    int Nwarm = 5;
    for(int n=0;n<Nwarm;++n){ //warmup
      mfma16Kernel(N, 8, a, o_x, o_y);
    }
    device.finish();

    //test
    for(int k=1;k<=256;k*=2){

      timePoint_t start = std::chrono::high_resolution_clock::now();;

      /* COPY Test */
      int Ntests = 20;
      for(int n=0;n<Ntests;++n){
        mfma16Kernel(N, k, a, o_x, o_y);
      }

      device.finish();
      timePoint_t end = std::chrono::high_resolution_clock::now();;
      double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/(1.0e6)/Ntests;

      size_t bytesIn  = N*sizeof(T);
      size_t bytesOut = N*sizeof(T);
      size_t bytes = bytesIn + bytesOut;

      size_t flops = (2048/64)*k*static_cast<size_t>(N);

      printf("MFMA16:        BW=%8.2f GB/s, AI=%8.2f FLOP/B, GFLOPS=%8.2f\n",
              (static_cast<double>(bytes)/1.e9)/elapsedTime, static_cast<double>(flops)/bytes, (static_cast<double>(flops)/1.e9)/elapsedTime);
    }
  }

  if (hasMFMAs) {
    int Nwarm = 5;
    for(int n=0;n<Nwarm;++n){ //warmup
      shmem1mfma16Kernel(N, 8, a, o_x, o_y);
    }
    device.finish();

    //test
    for(int k=1;k<=256;k*=2){
      timePoint_t start = std::chrono::high_resolution_clock::now();;

      /* COPY Test */
      int Ntests = 20;
      for(int n=0;n<Ntests;++n){
        shmem1mfma16Kernel(N, k, a, o_x, o_y);
      }

      device.finish();
      timePoint_t end = std::chrono::high_resolution_clock::now();;
      double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/(1.0e6)/Ntests;

      size_t bytesIn  = N*sizeof(T);
      size_t bytesOut = N*sizeof(T);
      size_t bytes = bytesIn + bytesOut;

      size_t flops = (2048/64)*k*static_cast<size_t>(N);

      printf("MFMA16 SHMEM1: BW=%8.2f GB/s, AI=%8.2f FLOP/B, GFLOPS=%8.2f, SHMEM AI=%8.2f FLOP/B\n",
              (static_cast<double>(bytes)/1.e9)/elapsedTime, static_cast<double>(flops)/bytes, (static_cast<double>(flops)/1.e9)/elapsedTime, static_cast<double>(2048)/(64*sizeof(T)));
    }
  }

  if (hasMFMAs) {
    int Nwarm = 5;
    for(int n=0;n<Nwarm;++n){ //warmup
      shmem2mfma16Kernel(N, 8, a, o_x, o_y);
    }
    device.finish();

    //test
    for(int k=1;k<=256;k*=2){
      timePoint_t start = std::chrono::high_resolution_clock::now();;

      /* COPY Test */
      int Ntests = 20;
      for(int n=0;n<Ntests;++n){
        shmem2mfma16Kernel(N, k, a, o_x, o_y);
      }

      device.finish();
      timePoint_t end = std::chrono::high_resolution_clock::now();;
      double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/(1.0e6)/Ntests;

      size_t bytesIn  = N*sizeof(T);
      size_t bytesOut = N*sizeof(T);
      size_t bytes = bytesIn + bytesOut;

      size_t flops = (2048/64)*k*static_cast<size_t>(N);

      printf("MFMA16 SHMEM2: BW=%8.2f GB/s, AI=%8.2f FLOP/B, GFLOPS=%8.2f, SHMEM AI=%8.2f FLOP/B\n",
              (static_cast<double>(bytes)/1.e9)/elapsedTime, static_cast<double>(flops)/bytes, (static_cast<double>(flops)/1.e9)/elapsedTime, static_cast<double>(2048)/(2*64*sizeof(T)));
    }
  }
}

int main(int argc, const char **argv) {

  // Parse arguments to json
  occa::json args = parseArgs(argc, argv);

  if (args["options/type"]=="fp32") {
    run<float>(args);
  } else {
    run<double>(args);
  }
}
