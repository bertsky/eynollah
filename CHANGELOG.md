Change Log
==========

Versioned according to [Semantic Versioning](http://semver.org/).

## Unreleased

## [0.1.0] - 2023-03-22

Fixed:

  * Do not produce spurious `TextEquiv`, #68
  * Less spammy logging, #64, #65, #71

Changed:

  * Upgrade to tensorflow 2.4.0, #74
  * Improved README
  * CI: test for python 3.7+, #90

## [0.0.11] - 2022-02-02

Fixed:

  * `models` parameter should have `content-type`, #61, OCR-D/core#777

## [0.0.10] - 2021-09-27

Fixed:

  * call to `uild_pagexml_no_full_layout` for empty pages, #52

## [0.0.9] - 2021-08-16

Added:

  * Table detection, #48

Fixed:

  * Catch exception, #47

## [0.0.8] - 2021-07-27

Fixed:

  * `pc:PcGts/@pcGtsId` was not set, #49

## [0.0.7] - 2021-07-01

Fixed:

  * `slopes`/`slopes_h` retval/arguments mixed up, #45, #46

## [0.0.6] - 2021-06-22

Fixed:

  * Cast arguments to opencv2 to python native types, #43, #44, opencv/opencv#20186

## [0.0.5] - 2021-05-19

Changed:

  * Remove `allow_enhancement` parameter, #42

## [0.0.4] - 2021-05-18

  * fix contour bug, #40

## [0.0.3] - 2021-05-11

  * fix NaN bug, #38

## [0.0.2] - 2021-05-04

Fixed:

  * prevent negative coordinates for textlines in marginals
  * fix a bug in the contour logic, #38
  * the binarization model is added into the models and now binarization of input can be done at the first stage of eynollah's pipline. This option can be turned on by -ib (-input_binary) argument. This is suggested for very dark or bright documents

## [0.0.1] - 2021-04-22

Initial release

<!-- link-labels -->
[0.1.0]: ../../compare/v0.1.0...v0.0.11
[0.0.11]: ../../compare/v0.0.11...v0.0.10
[0.0.10]: ../../compare/v0.0.10...v0.0.9
[0.0.9]: ../../compare/v0.0.9...v0.0.8
[0.0.8]: ../../compare/v0.0.8...v0.0.7
[0.0.7]: ../../compare/v0.0.7...v0.0.6
[0.0.6]: ../../compare/v0.0.6...v0.0.5
[0.0.5]: ../../compare/v0.0.5...v0.0.4
[0.0.4]: ../../compare/v0.0.4...v0.0.3
[0.0.3]: ../../compare/v0.0.3...v0.0.2
[0.0.2]: ../../compare/v0.0.2...v0.0.1
[0.0.1]: ../../compare/HEAD...v0.0.1
