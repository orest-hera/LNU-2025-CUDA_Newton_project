execute_process(
    COMMAND git describe --tags --always --dirty
    WORKING_DIRECTORY ${SOURCE_DIR}
    OUTPUT_VARIABLE GIT_VERSION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

message(STATUS "Version: ${GIT_VERSION_STRING}")

configure_file(
    ${SOURCE_DIR}/version.h.in
    ${BINARY_DIR}/version.h
    @ONLY
)
