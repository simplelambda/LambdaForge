"""Allow ``python -m lambdaforge`` to invoke the CLI."""

from lambdaforge.cli.CommandLineInterface import CommandLineInterface

raise SystemExit(CommandLineInterface.main())
