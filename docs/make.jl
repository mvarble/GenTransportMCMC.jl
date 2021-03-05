using Documenter, GenTransportMCMC

makedocs(
  sitename="GenTransportMCMC",
  modules=[GenTransportMCMC],
  pages=[
    "Home" => "index.md",
    "Getting Started" => "getting_started.md",
    "Mathematical Details" => "mathematical_details.md",
    "API" => [
      "Transport" => "ref/transport.md",
    ],
  ],
)

deploydocs(
  repo = "github.com/mvarble/gen-transport-mcmc.git",
  target = "build"
)